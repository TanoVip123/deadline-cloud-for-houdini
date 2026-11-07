# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""``PdgHost`` over live PDG and Houdini.

Only this module and the registration module import ``hou`` or ``pdg``. PDG calls
the scheduler on its own threads, and ``hou`` is not safe there while Houdini's
UI runs, so every scene read runs on the main thread.
"""

from __future__ import annotations

import dataclasses
import glob
import logging
import os
import re
import threading
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

import hou  # type: ignore[import-not-found]
import pdg  # type: ignore[import-not-found]

from ..errors import CookSetupError, PayloadTransferError
from ..graph import GraphSnapshot, NodeInfo, synthesize
from ..ports import ItemOutcome, ItemState, SceneFiles, WorkItemHandle
from ..scene_reads import ReadReport, SceneReads, match_reads
from ..validation import Refusal, RefusalScenario, ValidationContext, Verdict, validate, waived
from . import readers
from .settings import SchedulerSettings

_logger: logging.Logger = logging.getLogger(__name__)

#: What a read run on the main thread returns.
T = TypeVar("T")

#: The scheduler's node type, which the HDA defines.
_NODE_TYPE: str = "deadlinecloudscheduler"

_WORKING_DIR_PARM: str = "working_dir"
_WORKING_DIR_DEFAULT: str = "$HIP"
_DEBUG_LEVEL_PARM: str = "debug_level"
_VALIDATE_GRAPH_PARM: str = "validate_graph"
#: 0 is the standard record, and 1 adds the graph dump and the item details.
_DEBUG_LEVELS: Tuple[int, ...] = (0, 1)
#: Wins over the parm, so a scene saved before the parm existed can still be debugged.
_DEBUG_LEVEL_ENV: str = "DEADLINE_PDG_DEBUG_LEVEL"
_AUTO_SAVE_PARM: str = "auto_save_hip"
#: The save prompt's buttons, in the order ``hou.ui.displayMessage`` numbers them.
_SAVE_BUTTONS: Tuple[str, ...] = ("Save", "Always Save", "Cancel")

#: PDG's sub-index for an item that is not in a batch. The cook refuses batches.
_NOT_BATCHED: int = -1

#: PDG reads an output's checksum as its change stamp. The file was written on a
#: worker, so there is nothing to stamp here, and 0 is PDG's own default.
_NO_CHECKSUM: int = 0

#: The findings that need generated work items, so a verdict taken before
#: generation would report every step.
_NEEDS_ITEMS: FrozenSet[RefusalScenario] = frozenset({RefusalScenario.STEP_WITHOUT_ITEMS})

_SAVE_REMEDY: str = "Save the scene, then cook again."

#: The environment variables that can name the home directory. Windows sets
#: ``USERPROFILE``, and a shell on any system may set ``HOME``.
_HOME_VARIABLES: Tuple[str, ...] = ("HOME", "USERPROFILE")

# ``pdg.workItemState`` names of an item PDG has queued in the current cook.
_QUEUED_STATES: FrozenSet[str] = frozenset({"Waiting", "Scheduled", "Cooking"})

# (node path, parm) pairs already reported missing, so an old node logs each once.
_reported_missing: Set[Tuple[str, str]] = set()
_reported_lock: threading.Lock = threading.Lock()


class HoudiniPdgHost:
    """Implements ``PdgHost`` for one scheduler node."""

    def __init__(self, scheduler: "pdg.scheduler.PyScheduler") -> None:
        self._scheduler: Any = scheduler
        self._lock: threading.Lock = threading.Lock()
        # One host serves one cook, and the cook and the submitter both ask.
        self._scene_files: Optional[SceneFiles] = None
        self._node: Any = None
        # The steps a cook waits for, and those PDG has finished generating.
        self._waiting_for: Set[str] = set()
        self._generated: Set[str] = set()
        self._generation_handler: Any = None
        # Separate from ``_lock``, which is held across a main-thread call.
        self._generation_lock: threading.Lock = threading.Lock()

    def read_graph(self) -> Sequence[NodeInfo]:
        """The topology and the generated items, from one ``dependencyGraph`` read."""
        dependency_graph: Any = self._scheduler.dependencyGraph(True)
        return self.on_main_thread(lambda: self._read_graph(dependency_graph))

    def read_scene(self) -> SceneReads:
        """The scene reads of every TOP node, so a node that is not a step is judged too."""
        return self.on_main_thread(
            lambda: readers.collect_scene_reads(
                _top_nodes(_graph_nodes(self._scheduler.context)), expand=_expand_variables
            )
        )

    def scene_files(self) -> SceneFiles:
        """The saved ``.hip`` and the assets the ROP submitter's scan finds.

        Raises ``CookSetupError`` for a path that would attach a whole filesystem
        root or home directory.
        """
        with self._lock:
            if self._scene_files is None:
                node: Any = self.scheduler_node()
                self._scene_files = self.on_main_thread(lambda: _scene_files(node))
            return self._scene_files

    def serialize(self, item: WorkItemHandle) -> str:
        """PDG's own JSON for the work item."""
        if item.native is None:
            raise PayloadTransferError(f"Work item '{item.name}' has no PDG object to serialize.")
        # What PDG's own job RPC returns for an item.
        return str(item.native.saveJSONString(global_attributes=True))

    def notify(self, item: WorkItemHandle, outcome: ItemOutcome) -> None:
        """Report the outcome through PDG's ``onWorkItem*`` callbacks."""
        scheduler: Any = self._scheduler
        item_id: int = item.item_id
        if outcome.state is ItemState.STARTED:
            # The start flag leaves an earlier cook's output records in place, and
            # PDG adds this cook's beside them, so the item is cleared first.
            _clear_outputs(item)
            scheduler.onWorkItemStartCook(item_id, _NOT_BATCHED, True)
        elif outcome.state is ItemState.SUCCEEDED:
            # Each output is reported unchanged, then the success with its duration.
            for output in outcome.outputs:
                scheduler.onWorkItemAddOutput(
                    item_id, _NOT_BATCHED, output.path, output.tag, _NO_CHECKSUM
                )
            scheduler.onWorkItemSucceeded(item_id, _NOT_BATCHED, float(outcome.duration_seconds))
        else:
            scheduler.onWorkItemFailed(item_id, _NOT_BATCHED)

    def set_working_dir(self, path: str) -> None:
        """Use ``path`` as both the local and the remote working directory."""
        self._scheduler.setWorkingDir(path, path)

    def watch_generation(self, cook_set: Optional[AbstractSet[str]]) -> None:
        """Follow ``NodeGenerated`` for the steps PDG asked for.

        PDG offers a step's first item while it is still generating the steps
        below it, so a graph read at that moment is missing most of the job.
        """
        context: Any = self._scheduler.context
        steps: Set[str] = set(_step_top_nodes(_graph_nodes(context)))
        with self._generation_lock:
            self._waiting_for = steps & set(cook_set) if cook_set else steps
            self._generated = set()
        handler: Any = context.addEventHandler(
            self._on_node_generated, pdg.EventType.NodeGenerated, True
        )
        with self._generation_lock:
            self._generation_handler = handler

    def pending_generation(self) -> Sequence[str]:
        """The watched steps that have neither generated nor queued an item."""
        with self._generation_lock:
            pending: Set[str] = self._waiting_for - self._generated
        if not pending:
            return ()
        # A node generated before this cook may not announce itself again. Once
        # PDG queues one of its items, its generation is done.
        started: Set[str] = {
            _name(node)
            for node in _graph_nodes(self._scheduler.context)
            if _name(node) in pending and _has_queued_item(node)
        }
        with self._generation_lock:
            self._generated |= started
            return tuple(sorted(self._waiting_for - self._generated))

    def stop_watching_generation(self) -> None:
        """Remove the event handler, if there is one."""
        with self._generation_lock:
            handler: Any = self._generation_handler
            self._generation_handler = None
        if handler is not None:
            try:
                self._scheduler.context.removeEventHandler(handler)
            except Exception:
                _logger.debug("Could not remove the generation event handler", exc_info=True)

    def failure_reason(self, item_id: int) -> Tuple[str, str]:
        """The ERROR lines of the work item's PDG log, and its last line.

        The last line is kept apart, because without an ERROR line it may be an
        unrelated message.
        """
        try:
            graph: Any = _attr(self._scheduler.context, "graph")
            item: Any = _call(graph, "workItemById", item_id)
            return _reason_from_log(str(_attr(item, "logMessages") or ""))
        except Exception:
            _logger.debug("Could not read the log of work item %s.", item_id, exc_info=True)
            return "", ""

    def houdini_version(self) -> str:
        """The running Houdini's version, computed as the ROP computes it."""
        from deadline.houdini_submitter.python.deadline_cloud_for_houdini.submitter import (
            _get_houdini_version,
        )

        return self.on_main_thread(_get_houdini_version)

    def queue_parameter_values(self) -> List[Dict[str, Any]]:
        """The scheduler node's queue parameter values, read as the ROP reads them."""
        from deadline.houdini_submitter.python.deadline_cloud_for_houdini.queue_parameters import (
            get_queue_parameter_values_as_openjd,
        )

        node: Any = self.scheduler_node()
        return self.on_main_thread(lambda: get_queue_parameter_values_as_openjd(node))

    def on_main_thread(self, read: Callable[[], T]) -> T:
        """Run ``read`` on Houdini's main thread whenever a UI exists, and call it directly otherwise."""
        return on_main_thread(read)

    def scheduler_node(self) -> Any:
        """The scheduler's own TOP node, which holds the cook's parms."""
        if self._node is None:
            self._node = self.on_main_thread(lambda: _scheduler_node(self._scheduler))
        return self._node

    def read_settings(self) -> SchedulerSettings:
        """The cook's settings, from the scheduler node's parms."""
        node: Any = self.scheduler_node()
        return self.on_main_thread(lambda: read_scheduler_settings(node))

    def _on_node_generated(self, _handler: Any, event: Any) -> None:
        # Record the node PDG says finished generating.
        name: str = _name(_attr(event, "node"))
        if name:
            with self._generation_lock:
                self._generated.add(name)

    def _read_graph(self, dependency_graph: Any) -> Sequence[NodeInfo]:
        # The nodes and their items, with an override naming this scheduler cleared.
        context: Any = self._scheduler.context
        graph_nodes: List[Any] = _graph_nodes(context)
        infos: Tuple[NodeInfo, ...] = readers.node_infos(
            graph_nodes,
            dependency_graph,
            cook_set_names(_attr(context, "cookSet")),
            only_items_in_cook=True,
        )
        return _own_overrides_cleared(infos, graph_nodes, self._own_names())

    def _own_names(self) -> FrozenSet[str]:
        # Every name an override may use for this scheduler.
        names: Set[str] = {_name(self._scheduler)}
        try:
            names |= _names_of(self.scheduler_node())
        except CookSetupError:
            pass
        return frozenset(name for name in names if name)


class PreCookChecks:
    """The checks that run before a cook starts, from the TOP Cook button or the node's Submit.

    The unsaved ``.hip`` prompt first, then the refusals the topology alone can
    show. With no UI, the prompt becomes a refusal.
    """

    def run(self, node: Any, *, interactive: bool) -> Verdict:
        """The verdict for starting a cook on the scheduler ``node``. A refusal or a cancel starts nothing.

        With ``interactive``, unsaved changes are saved first, or the artist cancels,
        which raises ``UserInitiatedCancel``.
        """
        if interactive:
            on_main_thread(lambda: _prompt_to_save(node))
        verdict: Verdict = on_main_thread(lambda: _topology_verdict(node))
        refusals: Tuple[Refusal, ...] = verdict.refusals + on_main_thread(_save_refusals)
        return Verdict(refusals=refusals, advisories=verdict.advisories)


def on_main_thread(read: Callable[[], T]) -> T:
    """Call ``read`` on Houdini's main thread, and return its result or raise its error.

    With no UI, as in ``hython``, nothing races, so it is called directly.
    """
    if not hou.isUIAvailable() or threading.current_thread() is threading.main_thread():
        return read()
    import hdefereval  # type: ignore[import-not-found]

    result: T = hdefereval.executeInMainThreadWithResult(read)
    return result


def cook_set_names(cook_set: Any) -> Optional[FrozenSet[str]]:
    """PDG's cook set, a list of ``pdg.Node``, as node names. None when unknown."""
    try:
        names: FrozenSet[str] = frozenset(_name(node) or str(node) for node in (cook_set or ()))
    except Exception:
        return None
    # An empty set would leave every node out, which is not what PDG asked.
    return names or None


def find_scheduler_node(scheduler: Any) -> Any:
    """The TOP node of ``scheduler``, read on the main thread. Raises ``CookSetupError`` when there is none."""
    return on_main_thread(lambda: _scheduler_node(scheduler))


def _clear_outputs(item: WorkItemHandle) -> None:
    """Drop the output records an earlier cook left on the work item.

    An item with no PDG object, or a PDG build whose item refuses the call, keeps
    its records, because raising here would lose the task.
    """
    native: Any = item.native
    if native is None:
        return
    try:
        native.clearOutputFiles()
    except Exception:
        _logger.warning("Could not clear the earlier outputs of '%s'.", item.name, exc_info=True)


# -- settings -----------------------------------------------------------------


def read_scheduler_settings(node: Any) -> SchedulerSettings:
    """The cook's settings from the scheduler node's parms. An absent parm takes its default.

    Call it on the main thread.
    """
    # The working directory must be a real, absolute path.
    working_dir: str = _absolute(_parm(node, _WORKING_DIR_PARM, _WORKING_DIR_DEFAULT, as_text=True))
    if not working_dir:
        raise CookSetupError(
            f"The '{_WORKING_DIR_PARM}' parm is empty. Set it to the directory the cook "
            "writes to, usually $HIP."
        )
    if "$" in working_dir:
        raise CookSetupError(
            f"The '{_WORKING_DIR_PARM}' parm names a variable that is not set: "
            f"'{working_dir}'. Use a real path, or $HIP."
        )
    return SchedulerSettings(
        working_dir=working_dir,
        debug_level=_debug_level(node),
        validate_graph=bool(_parm(node, _VALIDATE_GRAPH_PARM, 1, as_text=False)),
    )


def _debug_level(node: Any) -> int:
    """The debug level: the environment variable when it is valid, else the parm."""
    override: str = os.environ.get(_DEBUG_LEVEL_ENV, "").strip()
    if override:
        level: int = int(override) if re.fullmatch(r"-?\d+", override) else -1
        if level in _DEBUG_LEVELS:
            return level
        _logger.warning(
            "Ignoring %s=%r: it must be 0 or 1.",
            _DEBUG_LEVEL_ENV,
            override,
        )
    # The parm holds a menu index, so anything else is a damaged scene.
    value: Any = _parm(node, _DEBUG_LEVEL_PARM, 0, as_text=False)
    try:
        parm_level: int = int(value)
    except (TypeError, ValueError):
        parm_level = -1
    if parm_level not in _DEBUG_LEVELS:
        raise CookSetupError(f"The '{_DEBUG_LEVEL_PARM}' parm must be 0 or 1, not {value!r}.")
    return parm_level


def _parm(node: Any, name: str, default: Any, *, as_text: bool) -> Any:
    """The parm's value, or ``default`` when the node lacks it. An unreadable parm raises ``CookSetupError``."""
    parm: Any = node.parm(name)
    if parm is None:
        _report_missing(node, name, default)
        return hou.text.expandString(default) if as_text else default
    try:
        return parm.evalAsString() if as_text else parm.eval()
    except Exception as e:
        raise CookSetupError(f"The '{name}' parm cannot be read: {e}") from e


def _report_missing(node: Any, name: str, default: Any) -> None:
    """Log once per node and parm that the parm is missing, and which default is used."""
    key: Tuple[str, str] = (_path_of(node), name)
    with _reported_lock:
        if key in _reported_missing:
            return
        _reported_missing.add(key)
    _logger.warning(
        "Node %s has no '%s' parm, so it uses the default, %r. Recreate the node to get it.",
        key[0] or "?",
        name,
        default,
    )


def _depends_on_work_item(raw: str) -> bool:
    """Whether an authored value is an expression or names a work item attribute, such as `` `@pdg_input` ``."""
    return "`" in raw or "@" in raw


def _absolute(path: str) -> str:
    """``path`` made absolute and normalized, or ``""`` for an empty path."""
    stripped: str = (path or "").strip()
    return os.path.normpath(os.path.abspath(os.path.expanduser(stripped))) if stripped else ""


# -- the graph ----------------------------------------------------------------


def _scheduler_node(scheduler: Any) -> Any:
    """The scheduler's TOP node: PDG's own link first, then the only node of this type by name."""
    try:
        node: Any = scheduler.topNode()
    except Exception:
        node = None
    if node is not None:
        return node
    # PDG gave no link, so match the node by the scheduler's name.
    name: str = _name(scheduler)
    try:
        instances: List[Any] = list(hou.nodeType(hou.topNodeTypeCategory(), _NODE_TYPE).instances())
    except Exception:
        instances = []
    matches: List[Any] = [n for n in instances if _call(n, "name") == name]
    if not matches and len(instances) == 1:
        matches = instances
    if len(matches) == 1:
        return matches[0]
    raise CookSetupError(
        f"Cannot find the node of scheduler '{name}' in the scene, so its settings cannot be "
        "read. Recreate the scheduler node."
    )


# A PDG log line's leading "[HH:MM:SS.fff] " stamp.
_LOG_STAMP: "re.Pattern[str]" = re.compile(r"^\[[0-9:.]+\]\s*")


def _reason_from_log(log: str) -> Tuple[str, str]:
    """A work item's log as ``(its ERROR: lines joined, its last line)``, without time stamps."""
    lines: List[str] = [_LOG_STAMP.sub("", line).strip() for line in log.splitlines()]
    lines = [line for line in lines if line]
    errors: List[str] = [line.split("ERROR:", 1)[1].strip() for line in lines if "ERROR:" in line]
    return " ".join(errors), (lines[-1] if lines else "")


def _graph_nodes(context: Any) -> List[Any]:
    """The PDG nodes of a graph context, or none."""
    graph: Any = _attr(context, "graph")
    nodes: Any = _call(graph, "nodes") if graph is not None else None
    return list(nodes or ())


# ``$NAME`` or ``${NAME}``.
_VARIABLE: "re.Pattern[str]" = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _expand_variables(text: str) -> str:
    """``text`` with Houdini's variables expanded. Raises for a variable Houdini does not define.

    Houdini expands an unknown variable to nothing, which would shorten a path.
    """
    for braced, bare in _VARIABLE.findall(text):
        name: str = braced or bare
        if hou.getenv(name) is None:
            raise KeyError(name)
    return str(hou.text.expandString(text))


def _has_queued_item(node: Any) -> bool:
    """Whether PDG has queued one of the node's items in this cook."""
    for item in _attr(node, "workItems") or ():
        if readers._enum_name(_attr(item, "state")) in _QUEUED_STATES:
            return True
    return False


def _names_of(node: Any) -> Set[str]:
    """The node's name, path, and PDG node name, any of which an override may hold."""
    names: Set[str] = {str(_call(node, "name") or ""), _path_of(node)}
    pdg_name: Any = _call(node, "getPDGNodeName")
    if pdg_name:
        names.add(str(pdg_name))
    return {name for name in names if name}


def _own_overrides_cleared(
    infos: Sequence[NodeInfo], graph_nodes: Iterable[Any], own: Iterable[str]
) -> Tuple[NodeInfo, ...]:
    """Clear an override that names this scheduler, because it changes nothing.

    The parm holds a path as authored, often relative, so PDG's own answer for
    the node's scheduler decides when it can be read.
    """
    own_names: Set[str] = set(own)
    # PDG's resolved scheduler for each node, by node name.
    resolved: Dict[str, str] = {}
    for node in graph_nodes:
        name: str = _name(node)
        scheduler: str = _name(_attr(node, "scheduler"))
        if name and scheduler:
            resolved[name] = scheduler
    return tuple(
        (
            dataclasses.replace(info, scheduler_override="")
            if info.scheduler_override
            and (info.scheduler_override in own_names or resolved.get(info.name) in own_names)
            else info
        )
        for info in infos
    )


def _is_step(node: Any) -> bool:
    """Whether a PDG node runs a job script, which makes it a step."""
    try:
        return bool(node.scriptInfo.command)
    except Exception:
        return False


def _step_top_nodes(graph_nodes: Iterable[Any]) -> Dict[str, Any]:
    """Each step's ``hou.TopNode``, by PDG node name."""
    return {
        _name(node): _call(node, "topNode")
        for node in graph_nodes
        if _name(node) and _is_step(node)
    }


def _top_nodes(graph_nodes: Iterable[Any]) -> Dict[str, Any]:
    """Every PDG node's ``hou.TopNode``, by PDG node name, steps or not."""
    return {_name(node): _call(node, "topNode") for node in graph_nodes if _name(node)}


# -- scene files --------------------------------------------------------------


def _scene_files(node: Any) -> SceneFiles:
    """The saved ``.hip`` and the scene's assets. Raises ``CookSetupError`` for an unbounded path."""
    hip: str = str(hou.hipFile.path() or "")
    hip_file: str = _absolute(hip) if hip else ""
    found: Dict[str, Set[str]] = _scene_assets(node)
    paths: Set[str] = set(found) - {hip_file} - _integration_libraries()
    _refuse_unbounded(paths, found, os.environ)
    # Job attachments rejects a directory listed as a file. An asset library can
    # be stored expanded, as a directory.
    directories: Set[str] = {path for path in paths if os.path.isdir(path)}
    return SceneFiles(
        hip_file=hip_file,
        assets=tuple(sorted(paths - directories)),
        directories=tuple(sorted(directories)),
    )


# The node types of this integration. A worker never cooks them, and their
# libraries are installed with the submitter, not with the scene.
_INTEGRATION_NODE_TYPES: Tuple[Tuple[str, str], ...] = (
    ("topNodeTypeCategory", "deadlinecloudscheduler"),
    ("ropNodeTypeCategory", "deadline_cloud"),
)


def _integration_libraries() -> Set[str]:
    """The asset libraries that define this integration's own node types."""
    libraries: Set[str] = set()
    for category, name in _INTEGRATION_NODE_TYPES:
        try:
            node_type: Any = hou.nodeType(getattr(hou, category)(), name)
            definition: Any = node_type.definition() if node_type is not None else None
            if definition is not None:
                libraries.add(_absolute(str(definition.libraryFilePath())))
        except Exception:
            _logger.debug("Could not find the library of node type '%s'.", name, exc_info=True)
    return libraries


def _scene_assets(node: Any) -> Dict[str, Set[str]]:
    """The ROP's scene scan, as absolute paths that exist, each with the parms that name it.

    The scan returns references as authored, with ``$HIP`` and ``$F`` intact. A
    reference that holds an expression or a work item attribute, such as
    `` `@pdg_input` ``, is left out, because the step that writes it delivers it.
    """
    try:
        from deadline.houdini_submitter.python.deadline_cloud_for_houdini._assets import (
            _IGNORE_REF_VALUES,
            _get_scene_asset_references,
            _houdini_time_vars_to_glob,
        )

        refs: Any = _get_scene_asset_references(node)
        candidates: Set[str] = set(refs.input_filenames) | set(refs.input_directories)
        # The parms that hold each reference. An installed asset library is a
        # reference that no parm holds, and the shared scan skips those.
        owners: Dict[str, Set[str]] = {}
        for parm, ref in hou.fileReferences():
            if not ref or str(ref).startswith(_IGNORE_REF_VALUES):
                continue
            if parm is None:
                candidates.add(str(ref))
            owners.setdefault(str(ref), set()).add(
                _path_of(parm) if parm is not None else "an asset library"
            )
    except Exception:
        _logger.warning(
            "The scene's assets could not be scanned, so files the scene references by "
            "path may be missing on the workers.",
            exc_info=True,
        )
        return {}

    assets: Dict[str, Set[str]] = {}
    for ref in sorted(candidates):
        if not ref:
            continue
        if _depends_on_work_item(ref):
            _logger.info(
                "Not attaching '%s': it depends on a work item, so the step that writes "
                "the file delivers it.",
                ref,
            )
            continue
        # Expand the variables, and keep only an absolute path with none left. A
        # relative path would resolve against Houdini's current directory.
        pattern: str = _houdini_time_vars_to_glob(ref)
        expanded: str = os.path.expanduser(hou.text.expandString(pattern).strip())
        if not os.path.isabs(expanded) or "$" in expanded:
            continue
        expanded = _absolute(expanded)
        matched: List[str] = (
            [_absolute(path) for path in glob.glob(expanded)]
            if pattern != ref
            else ([expanded] if os.path.exists(expanded) else [])
        )
        for path in matched:
            assets.setdefault(path, set()).update(owners.get(ref, {ref}))
    return assets


def _refuse_unbounded(
    paths: Iterable[str], owners: Mapping[str, Set[str]], env: Mapping[str, str]
) -> None:
    """Raise ``CookSetupError`` naming each path that is a filesystem root, a home directory, or above one.

    Attaching such a path would hash every file below it before anything is
    submitted, so the cook would hang with no message.
    """
    homes: List[str] = [
        os.path.normcase(_absolute(env[name])) for name in _HOME_VARIABLES if env.get(name)
    ]
    unbounded: List[str] = sorted(path for path in paths if _is_unbounded(path, homes))
    if not unbounded:
        return
    named: str = "; ".join(f"'{path}' from {', '.join(sorted(owners[path]))}" for path in unbounded)
    raise CookSetupError(
        f"The scene names a folder that would attach a whole drive or home directory: {named}. "
        "Point each parm at the folder that holds only what the scene needs, or save the "
        ".hip in its own project folder."
    )


def _is_unbounded(path: str, homes: Sequence[str]) -> bool:
    """Whether ``path`` is a filesystem root, one of ``homes``, or a directory that holds one."""
    folded: str = os.path.normcase(os.path.normpath(path))
    if os.path.dirname(folded) == folded:
        return True
    prefix: str = folded.rstrip(os.sep) + os.sep
    return any(home == folded or home.startswith(prefix) for home in homes)


# -- pre-cook checks ----------------------------------------------------------


def _prompt_to_save(node: Any) -> None:
    """Save unsaved changes before a submit: at once when the node's auto-save is on, else after asking.

    The prompt offers Save, Always Save, which also turns auto-save on, and Cancel,
    which raises ``UserInitiatedCancel``.
    """
    if not hou.isUIAvailable() or hou.hipFile.isNewFile() or not hou.hipFile.hasUnsavedChanges():
        # A scene that was never saved needs a file name, which this prompt cannot
        # ask for, so the refusal tells the artist to save it.
        return
    parm: Any = node.parm(_AUTO_SAVE_PARM)
    if not (parm is not None and parm.eval()):
        choice: int = hou.ui.displayMessage(
            "The scene has unsaved changes. The farm cooks the saved scene, so save it first.",
            title="Deadline Cloud",
            buttons=_SAVE_BUTTONS,
            default_choice=0,
            close_choice=_SAVE_BUTTONS.index("Cancel"),
        )
        if choice not in (0, 1):
            from deadline.client.exceptions import UserInitiatedCancel

            raise UserInitiatedCancel("Submission canceled.")
        # Always Save turns auto-save on, so the next submit saves without asking.
        if choice == 1 and parm is not None:
            parm.set(1)
    hou.hipFile.save()


def _save_refusals() -> Tuple[Refusal, ...]:
    """A refusal when the ``.hip`` was never saved, or, with a UI, has unsaved changes."""
    path: str = str(hou.hipFile.path() or "")
    detail: str
    if hou.hipFile.isNewFile():
        detail = "The scene has never been saved, so there is no .hip for the workers to load."
    # hython reports unsaved changes even right after a save, and a session with no
    # UI has no artist editing the scene, so only a graphical session is checked.
    elif hou.isUIAvailable() and hou.hipFile.hasUnsavedChanges():
        detail = (
            "The scene has unsaved changes. The workers load the saved .hip, so they would cook "
            "the scene as it was last saved, not as it is now."
        )
    else:
        return ()
    return (
        Refusal(
            scenario=RefusalScenario.UNSAVED_SCENE,
            subject=path or "the scene",
            detail=detail,
            remedy=_SAVE_REMEDY,
        ),
    )


def _topology_verdict(node: Any) -> Verdict:
    """The refusals the TOP network shows before PDG generates any work item."""
    context: Any = _graph_context(node)
    if context is None:
        _logger.warning(
            "The TOP network has no PDG graph yet, so the checks that need it run when the "
            "cook starts."
        )
        return Verdict()
    graph_nodes: List[Any] = _graph_nodes(context)
    infos: Tuple[NodeInfo, ...] = _own_overrides_cleared(
        readers.node_infos(graph_nodes, None), graph_nodes, _names_of(node)
    )
    snapshot: GraphSnapshot = synthesize(infos)
    # A settings problem is reported by the cook when it starts, so the topology
    # is still judged without it.
    settings: Optional[SchedulerSettings]
    try:
        settings = read_scheduler_settings(node)
    except CookSetupError:
        settings = None
    working_dir: str = settings.working_dir if settings is not None else ""
    scene: SceneReads = readers.collect_scene_reads(
        _top_nodes(graph_nodes), expand=_expand_variables
    )
    reads: Optional[ReadReport] = None
    if scene.available:
        # Only how the write parms are authored is known before generation. With no
        # declared outputs every read would look external, so matches are dropped.
        reads = dataclasses.replace(
            match_reads(snapshot, scene.reads, working_dir),
            matches=(),
            data_edges={},
            attribute_variables=(),
        )
    verdict: Verdict = validate(
        ValidationContext(
            nodes=infos,
            snapshot=snapshot,
            reads=reads,
            working_dir=working_dir,
            env=_environment(),
        )
    )
    # The checks that need generated items would report every step, so they wait.
    verdict = Verdict(
        refusals=tuple(r for r in verdict.refusals if r.scenario not in _NEEDS_ITEMS),
        advisories=tuple(a for a in verdict.advisories if a.scenario not in _NEEDS_ITEMS),
    )
    return waived(verdict) if settings is not None and not settings.validate_graph else verdict


def _graph_context(node: Any) -> Any:
    """The node's PDG graph context, creating the PDG nodes first when there is none."""
    context: Any = _call(node, "getPDGGraphContext")
    if context is None:
        # Creates the PDG nodes without generating or cooking anything.
        _call_kwargs(node, "cookWorkItems", block=True, tops_only=True)
        context = _call(node, "getPDGGraphContext")
    return context


def _environment() -> Dict[str, str]:
    """The process environment, with ``HIP`` taken from the saved scene when it is unset."""
    env: Dict[str, str] = dict(os.environ)
    if not env.get("HIP"):
        hip: str = str(hou.hipFile.path() or "")
        if hip:
            env["HIP"] = os.path.dirname(_absolute(hip))
    return env


# -- duck typing --------------------------------------------------------------


def _attr(obj: Any, name: str) -> Any:
    """``obj.name``, or None when it is absent or raises."""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _call(obj: Any, name: str, *args: Any) -> Any:
    """``obj.name(*args)``, or None when it is absent or raises."""
    try:
        return getattr(obj, name)(*args)
    except Exception:
        return None


def _call_kwargs(obj: Any, name: str, **kwargs: Any) -> Any:
    """``obj.name(**kwargs)``, or None when it is absent or raises, which is logged."""
    try:
        return getattr(obj, name)(**kwargs)
    except Exception:
        _logger.debug("%s failed", name, exc_info=True)
        return None


def _name(obj: Any) -> str:
    """A PDG object's ``name`` attribute, or ``""``."""
    return str(_attr(obj, "name") or "")


def _path_of(node: Any) -> str:
    """A HOM node's or parm's path, or ``""``."""
    return str(_call(node, "path") or "")
