# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The Houdini adaptor's actions for one PDG work item, which never branch on the node type.

A stock PDG job script cooks in the adaptor's own Houdini session, and any other command
runs through a shell. Only the parts that run in the client import Houdini's modules.
"""

from __future__ import annotations

# Only the standard library, so the module imports in Houdini's own Python.
import gzip
import importlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Tuple,
)

# In Houdini's Python the adaptor puts only the ``openjd`` and ``deadline`` namespace
# directories on the path, so the packages inside them import as top-level names there.
try:
    from openjd.adaptor_runtime_client import PathMappingRule

    from deadline.houdini_pdg.graph import PDG_DIR_TOKEN
    from deadline.houdini_pdg.ports import Clock, OutputFile, TaskAddress, TaskStorage
except ImportError:
    from adaptor_runtime_client import PathMappingRule  # type: ignore[import-not-found,no-redef]
    from houdini_pdg.graph import PDG_DIR_TOKEN  # type: ignore[import-not-found,no-redef]
    from houdini_pdg.ports import (  # type: ignore[import-not-found,no-redef]
        Clock,
        OutputFile,
        TaskAddress,
        TaskStorage,
    )

_logger: logging.Logger = logging.getLogger(__name__)

#: The client prints this, then the exit code, on a line of its own after every item.
#: The adaptor waits for it.
EXIT_CODE_MARKER: str = "PdgActions: work item exit code "

#: The variables the actions set for every item. An exported attribute never
#: overrides one, because PDG's own work item loader reads them.
_RESERVED_ENV: FrozenSet[str] = frozenset(
    {
        "PDG_ITEM_NAME",
        "PDG_ITEM_ID",
        "PDG_INDEX",
        "PDG_ITEM_DATA_SOURCE",
        "PDG_TEMP",
        "PDG_DIR",
        "PDG_OUTPUT_DIR",
        "PDG_SCRIPTDIR",
        "PDG_RESULT_SERVER",
    }
)

_TOKEN: Pattern[str] = re.compile(r"__PDG_[A-Z0-9_]+?__")

# The command PDG's job scripts are started with: `"__PDG_HYTHON__" "__PDG_SCRIPTDIR__/rop.py" json`.
_STOCK_COMMAND: Pattern[str] = re.compile(
    r'^\s*"?__PDG_HYTHON__"?\s+"?__PDG_SCRIPTDIR__/([A-Za-z_][A-Za-z0-9_]*)\.py"?\s+json\s*$'
)

_RESULT_PREFIX: str = "PDG_RESULT:"

# pdg.attribFlag, as pdgjob/pdgjson.py defines it. WorkItem.environment is the
# attributes with EnvExport and without Internal.
_ENV_EXPORT_FLAG: int = 0x2
_INTERNAL_FLAG: int = 0x8

# pdg.attribType. Values are formatted as PDG formats them, because a wedge value
# often ends up in a file name: a float 0 renders as "0.0", an int 1 as "1".
_ATTRIB_INT: int = 0
_ATTRIB_FLOAT: int = 1


@dataclass(frozen=True)
class WorkItemPayload:
    """A work item as PDG serialized it."""

    data: Mapping[str, Any]

    @property
    def _item(self) -> Mapping[str, Any]:
        block: Any = self.data.get("workitem")
        return block if isinstance(block, Mapping) else {}

    @property
    def item_name(self) -> str:
        """``workitem.name``, else ``{nodeName}_{id}``, else the top-level ``name``."""
        # PDG names an unnamed item after its node and ID, as its own loader does.
        item: Mapping[str, Any] = self._item
        if item.get("name"):
            return str(item["name"])
        node: Any = item.get("nodeName")
        item_id: Any = item.get("id")
        if node and item_id is not None:
            return f"{node}_{item_id}"
        return str(self.data.get("name") or "")

    @property
    def command(self) -> str:
        """``workitem.command``, else ``__pdg_commandstring[0]``, else ``command``."""
        item: Mapping[str, Any] = self._item
        if item.get("command"):
            return str(item["command"])
        # PDG keeps a node's command line in the first value of this attribute.
        attributes: Any = item.get("attributes")
        entry: Any = (
            attributes.get("__pdg_commandstring") if isinstance(attributes, Mapping) else None
        )
        values: Any = entry.get("value") if isinstance(entry, Mapping) else None
        if isinstance(values, list) and values and values[0]:
            return str(values[0])
        return str(self.data.get("command") or "")

    def environment(self) -> Dict[str, str]:
        """The attributes flagged ``EnvExport`` and not ``Internal``, formatted as PDG formats them.

        The payload's own ``environment`` field is always empty, so the attributes
        are the only source.
        """
        # A serialized environment block wins when it has anything in it.
        env: Dict[str, str] = {}
        for block in (self._item.get("environment"), self.data.get("environment")):
            if isinstance(block, Mapping):
                env.update({str(k): str(v) for k, v in block.items()})
        # Otherwise the exported attributes are the environment, as PDG builds it.
        if not env:
            attributes: Any = self._item.get("attributes")
            if isinstance(attributes, Mapping):
                for name, attrib in attributes.items():
                    value: Optional[str] = _exported_value(attrib)
                    if value is not None:
                        env[str(name)] = value
        return {k: v for k, v in env.items() if k not in _RESERVED_ENV and _is_settable(k, v)}

    def mapped(self, rules: Sequence[PathMappingRule]) -> "WorkItemPayload":
        """The payload with every string rewritten by the first rule that matches it."""
        if not rules:
            return self
        appliers: List[Tuple[str, Callable[..., Tuple[bool, str]]]] = [
            (str(getattr(rule, "source_path", "")), _applier(rule)) for rule in rules
        ]

        # A whole string maps by the first rule that matches it. A string no rule
        # matches may still hold paths inside it, such as a command line.
        def walk(value: Any) -> Any:
            if isinstance(value, str):
                for _source, apply in appliers:
                    changed: bool
                    mapped: str
                    changed, mapped = apply(path=value)
                    if changed:
                        return mapped
                return _map_embedded(value, appliers)
            if isinstance(value, Mapping):
                return {k: walk(v) for k, v in value.items()}
            if isinstance(value, list):
                return [walk(v) for v in value]
            return value

        return WorkItemPayload(walk(self.data))


class TokenResolver:
    """Replaces PDG's ``__PDG_*__`` tokens in a command with this worker's values.

    An unknown token, or the ID token with no ID, is left as is so the failure names it.
    ``__PDG_ITEM_LABEL__`` defaults to the item name, as a PDG work item's label does.
    """

    # The worker's platform. A subclass can set another one.
    _platform: str = sys.platform

    def __init__(
        self,
        *,
        hfs: str,
        pdg_dir: str,
        pdg_temp: str,
        item_name: str,
        python_version: str,
        item_id: str = "",
        item_label: str = "",
    ) -> None:
        # Houdini's own executables, or bare names that the shell finds on PATH.
        windows: bool = self._platform.startswith("win")
        hython: str = (
            (f"{hfs}/bin/hython.exe" if windows else f"{hfs}/bin/hython") if hfs else "hython"
        )
        python: str = _houdini_python(hfs, python_version, self._platform) if hfs else "python"
        scriptdir: str = f"{hfs}/houdini/python{python_version}libs/pdgjob" if hfs else ""
        self._values: Dict[str, str] = {
            "__PDG_HYTHON__": hython,
            "__PDG_PYTHON__": python,
            "__PDG_SCRIPTDIR__": scriptdir,
            "__PDG_HFS__": hfs,
            "__PDG_DIR__": pdg_dir,
            "__PDG_SHARED_ROOT__": pdg_dir,
            "__PDG_TEMP__": pdg_temp,
            "__PDG_SHARED_TEMP__": pdg_temp,
            "__PDG_ITEM_NAME__": item_name,
            "__PDG_ITEM_LABEL__": item_label or item_name,
            # There is no result server. The scheduler learns outcomes from the farm.
            "__PDG_RESULT_SERVER__": "",
        }
        if item_id:
            self._values["__PDG_ITEM_ID__"] = item_id

    def resolve(self, command: str) -> str:
        """The command with every known token replaced. ``__PDG_RESULT_SERVER__`` becomes empty."""
        return _TOKEN.sub(lambda m: self._values.get(m.group(0), m.group(0)), command)


@dataclass(frozen=True)
class _StagedItem:
    """A staged work item, as the client's two actions pass it from one to the next."""

    # The mapped payload, and the session's values the adaptor sent with it.
    payload: WorkItemPayload
    pdg_dir: str
    pdg_temp: str
    scene_file: str
    record_outputs: bool


class PdgActions:
    """The adaptor actions for PDG work items, keyed ``pdg_*`` so they never collide with the ROP's.

    The adaptor's process stages each payload and publishes its outputs. The Houdini client,
    which has no storage, runs the item with the session's environment plus the item's values.
    """

    def __init__(self, *, storage: Optional[TaskStorage] = None, clock: Clock) -> None:
        self._storage: Optional[TaskStorage] = storage
        self._clock: Clock = clock
        # The client's own interpreter is Houdini's, so this is Houdini's Python.
        self._python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}"
        self._baseline: Dict[str, str] = dict(os.environ)
        self._staged: Optional[_StagedItem] = None
        self._failure: str = ""
        # The staged item's address and the file its outputs come back in.
        self._pending: Optional[Tuple[TaskAddress, str]] = None

    @property
    def action_dict(self) -> Dict[str, Callable[[Dict[str, Any]], None]]:
        """The actions to merge into ``ClientInterface.actions``."""
        return {
            "pdg_set_environment": self.set_environment,
            "pdg_run_command": self.run_command,
        }

    def stage_payload(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch the payload, apply the session's path mapping, and write it where PDG's loader looks.

        Runs in the adaptor's process. Returns the client's arguments, and raises when the item can't run.
        """
        if self._storage is None:
            raise RuntimeError("Staging a work item needs storage, which only the adaptor has.")
        self._pending = None
        # The run data carries the task's IDs, because the daemon started before the task.
        address: TaskAddress = TaskAddress.from_env(
            {
                "DEADLINE_FARM_ID": str(args.get("farm_id") or ""),
                "DEADLINE_QUEUE_ID": str(args.get("queue_id") or ""),
                "DEADLINE_JOB_ID": str(args.get("job_id") or ""),
                "DEADLINE_STEP_ID": str(args.get("step_id") or ""),
                "DEADLINE_TASK_ID": str(args.get("task_id") or ""),
            }
        )
        rules: Sequence[PathMappingRule] = args.get("path_mapping_rules") or []
        payload: WorkItemPayload = WorkItemPayload(self._storage.get_payload(address)).mapped(rules)
        name: str = payload.item_name
        _check_item_name(name)
        # The session's PDG_TEMP wins, as it does for PDG's own schedulers.
        pdg_dir: str = str(args["pdg_dir"])
        pdg_temp: str = self._baseline.get("PDG_TEMP") or os.path.join(pdg_dir, "pdgtemp")
        os.makedirs(pdg_dir, exist_ok=True)
        os.makedirs(os.path.join(pdg_temp, "data"), exist_ok=True)
        path: str = _staged_path(pdg_temp, name)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload.data, f)
        # A previous attempt's outputs must not be published for this one.
        outputs_path: str = _outputs_path(pdg_temp, name)
        if os.path.lexists(outputs_path):
            os.remove(outputs_path)
        self._pending = (address, outputs_path)
        _logger.info("Staged work item '%s' at %s", name, path)
        return {
            "item_name": name,
            "pdg_dir": pdg_dir,
            "pdg_temp": pdg_temp,
            "scene_file": str(args.get("scene_file") or ""),
            "record_outputs": bool(args.get("record_outputs")),
        }

    def set_environment(self, args: Dict[str, Any]) -> None:
        """Set the ``PDG_*`` variables and the item's exported attributes."""
        self._restore_environment()
        self._staged, self._failure = None, ""
        # A failure here is reported by run_command, which always prints the exit code.
        try:
            # The name is checked again, because it becomes a file name here too.
            name: str = str(args.get("item_name") or "")
            _check_item_name(name)
            pdg_temp: str = str(args["pdg_temp"])
            with gzip.open(_staged_path(pdg_temp, name), "rt", encoding="utf-8") as f:
                payload: WorkItemPayload = WorkItemPayload(json.load(f))
            staged: _StagedItem = _StagedItem(
                payload=payload,
                pdg_dir=str(args["pdg_dir"]),
                pdg_temp=pdg_temp,
                scene_file=str(args.get("scene_file") or ""),
                record_outputs=bool(args.get("record_outputs")),
            )
            # The item's exported attributes, then PDG's own variables, which win.
            hfs: str = self._baseline.get("HFS", "")
            env: Dict[str, str] = payload.environment()
            env.update(
                {
                    "PDG_ITEM_NAME": payload.item_name,
                    "PDG_ITEM_DATA_SOURCE": "0",
                    "PDG_TEMP": pdg_temp,
                    "PDG_DIR": staged.pdg_dir,
                    "PDG_OUTPUT_DIR": staged.pdg_dir,
                    "PDG_SCRIPTDIR": (
                        f"{hfs}/houdini/python{self._python_version}libs/pdgjob" if hfs else ""
                    ),
                }
            )
            # PDG's own ID and index, which differ from the task's index on remapped steps.
            item: Mapping[str, Any] = payload._item
            for variable, key in (("PDG_ITEM_ID", "id"), ("PDG_INDEX", "index")):
                if item.get(key) is not None:
                    env[variable] = str(item[key])
            if hfs:
                env["HFS"] = hfs
            # The work item's hip attribute points into $PDG_DIR, so the scene is linked there.
            hip: str = _link_scene(staged.scene_file, staged.pdg_dir)
            if hip:
                env["HIP"] = os.path.dirname(hip)
            os.environ.update(env)
            os.environ.pop("PDG_RESULT_SERVER", None)
        except Exception as e:
            self._failure = f"Could not set the work item's environment: {e}"
            return
        self._staged = staged

    def run_command(self, args: Dict[str, Any]) -> None:
        """Run the command in ``$PDG_DIR``, streaming its output, and record its outputs for the adaptor when the step declares none."""
        # The environment is restored whatever happens, so nothing leaks into the next item.
        staged: Optional[_StagedItem] = self._staged
        failure: str = self._failure
        try:
            exit_code: int = self._run_or_report(staged, failure)
        finally:
            self._staged, self._failure = None, ""
            self._restore_environment()
        # The adaptor reads the code only at the start of a line, and a command's output
        # may not end with a newline, so the exit code line starts with one.
        print(f"\n{EXIT_CODE_MARKER}{exit_code}", flush=True)

    def publish_outputs(self) -> None:
        """Publish the outputs the client recorded for the staged item. Runs in the adaptor's process."""
        pending: Optional[Tuple[TaskAddress, str]] = self._pending
        self._pending = None
        if pending is None or self._storage is None:
            return
        address: TaskAddress
        path: str
        address, path = pending
        # The command already succeeded, so a failure here is logged and never fails the task.
        try:
            with open(path, encoding="utf-8") as f:
                entries: List[Dict[str, Any]] = json.load(f)["outputs"]
            outputs: List[OutputFile] = [
                OutputFile(str(e["path"]), str(e.get("tag") or "")) for e in entries
            ]
            self._storage.put_outputs(address, outputs)
            _logger.info("Recorded %d output(s) of the work item", len(outputs))
        except Exception as e:
            _logger.warning("Could not record the work item's outputs: %s", e)

    def _run_or_report(self, staged: Optional[_StagedItem], failure: str) -> int:
        """Run the staged item. An error is printed with its traceback and gives exit code 1."""
        try:
            return self._run(staged, failure)
        except Exception:
            # An escaped error would end the client and every later task, so only this item fails.
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            return 1

    def _run(self, staged: Optional[_StagedItem], failure: str) -> int:
        """Run the staged item and record its outputs. Returns its exit code."""
        if staged is None:
            print(f"ERROR: {failure or 'No work item was staged.'}", flush=True)
            return 1
        command: str = staged.payload.command
        if not command:
            print("ERROR: The work item payload carries no command.", flush=True)
            return 1
        started: datetime = self._clock.now()
        exit_code: int
        outputs: List[OutputFile]
        exit_code, outputs = self._execute(staged, command)
        # Only a step with no declared outputs records them. A silent command gets the scan.
        if staged.record_outputs and exit_code == 0:
            hip_link: str = os.path.join(staged.pdg_dir, os.path.basename(staged.scene_file))
            recorded: List[OutputFile] = outputs or _scan_outputs(
                staged.pdg_dir,
                exclude=(staged.pdg_temp, hip_link if staged.scene_file else ""),
                since=started,
            )
            _write_outputs(staged, recorded)
        return exit_code

    def _execute(self, staged: _StagedItem, command: str) -> Tuple[int, List[OutputFile]]:
        """Run the command. Returns its exit code and the outputs it reported."""
        # A stock job script cooks in this Houdini. Anything else runs through a shell.
        cook: Optional[Callable[[Any], Any]] = _stock_cook(command)
        if cook is not None:
            # A job script's cook either succeeds or fails, so its exit code is 0 or 1.
            print(f"PdgActions: cooking '{staged.payload.item_name}' in process", flush=True)
            succeeded: bool
            cooked: List[OutputFile]
            succeeded, cooked = _cook_in_process(cook, staged.pdg_dir)
            return (0 if succeeded else 1), cooked
        # A shell command gets this worker's values for PDG's tokens.
        resolved: str = TokenResolver(
            hfs=self._baseline.get("HFS", ""),
            pdg_dir=staged.pdg_dir,
            pdg_temp=staged.pdg_temp,
            item_name=staged.payload.item_name,
            python_version=self._python_version,
            item_id=_item_id(staged.payload),
            item_label=_item_label(staged.payload),
        ).resolve(command)
        print(f"PdgActions: running {resolved}", flush=True)
        exit_code: int
        lines: List[str]
        exit_code, lines = _run_shell(resolved, staged.pdg_dir)
        return exit_code, _parse_pdg_results(lines, staged.pdg_dir)

    def _restore_environment(self) -> None:
        """Put ``os.environ`` back to the session's baseline."""
        for key in [k for k in os.environ if k not in self._baseline]:
            del os.environ[key]
        for key, value in self._baseline.items():
            if os.environ.get(key) != value:
                os.environ[key] = value


def _item_id(payload: WorkItemPayload) -> str:
    """``workitem.id``, the number PDG's ``__PDG_ITEM_ID__`` and ``$PDG_ITEM_ID`` carry, or empty."""
    item_id: Any = payload._item.get("id")
    return "" if item_id is None or isinstance(item_id, bool) else str(item_id)


def _item_label(payload: WorkItemPayload) -> str:
    """The ``__pdg_label`` attribute, which is where PDG serializes a label that isn't the name."""
    # Each level may be missing or malformed, and then there is no label.
    attributes: Any = payload._item.get("attributes")
    entry: Any = attributes.get("__pdg_label") if isinstance(attributes, Mapping) else None
    values: Any = entry.get("value") if isinstance(entry, Mapping) else None
    if isinstance(values, list) and values and isinstance(values[0], str):
        return values[0]
    return ""


def _staged_path(pdg_temp: str, name: str) -> str:
    """Where ``pdgjson.getWorkItemJsonPath`` looks when ``PDG_ITEM_DATA_SOURCE`` is 0."""
    return os.path.join(pdg_temp, "data", f"{name}.json.gz")


def _outputs_path(pdg_temp: str, name: str) -> str:
    """Where the client hands the outputs over. Under ``$PDG_TEMP``, so the scan never reports it."""
    return os.path.join(pdg_temp, "data", f"{name}.outputs.json")


def _write_outputs(staged: _StagedItem, outputs: Sequence[OutputFile]) -> None:
    """Hand the outputs to the adaptor's process, with paths tokenized under ``__PDG_DIR__``."""
    record: List[Dict[str, str]] = [
        {"path": _tokenize(o.path, staged.pdg_dir), "tag": o.tag} for o in outputs
    ]
    path: str = _outputs_path(staged.pdg_temp, staged.payload.item_name)
    # An error fails the item, since a success with no record gives the next step wrong inputs.
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"outputs": record}, f)
    print(f"PdgActions: recorded {len(record)} output(s)", flush=True)


def _check_item_name(name: str) -> None:
    """Raise unless the name can be the file name of the item's staged payload."""
    if not name:
        raise ValueError("The work item payload carries no work item name.")
    if not _is_path_component(name):
        raise ValueError(f"The work item name {name!r} is not a single file name.")


def _exported_value(attrib: Any) -> Optional[str]:
    """The attribute's environment value, or None when PDG wouldn't export it."""
    if not isinstance(attrib, Mapping):
        return None
    # Only EnvExport without Internal is exported. A malformed flag exports nothing.
    try:
        flags: int = int(attrib.get("flag") or 0)
    except (TypeError, ValueError):
        return None
    if flags & _INTERNAL_FLAG or not flags & _ENV_EXPORT_FLAG:
        return None
    values: Any = attrib.get("value")
    if not isinstance(values, list) or not values:
        return None
    # A file attribute's values are objects, and they are never an environment value.
    parts: List[str] = []
    for value in values:
        if isinstance(value, (dict, list)):
            return None
        parts.append(_format_component(value, attrib.get("type")))
    # PDG joins a multi-component attribute with spaces, as hscript does.
    return " ".join(parts)


def _format_component(value: Any, attrib_type: Any) -> str:
    """One attribute value as PDG prints it, by the attribute's type."""
    try:
        if attrib_type == _ATTRIB_FLOAT:
            return str(float(value))
        if attrib_type == _ATTRIB_INT:
            return str(int(value))
    except (TypeError, ValueError, OverflowError):
        # A value that doesn't fit its type is exported as is, so malformed input never raises.
        pass
    return str(value)


def _is_settable(name: str, value: str) -> bool:
    """Whether ``os.environ`` accepts the name and value."""
    return bool(name) and "=" not in name and "\0" not in name and "\0" not in value


def _is_path_component(name: str) -> bool:
    """Whether the name is one file name on any platform, not a path."""
    return (
        name not in (".", "..")
        and "/" not in name
        and "\\" not in name
        and "\0" not in name
        and ":" not in name
    )


# A path embedded in a longer string, such as an ImageMagick argument list, starts
# after one of these, and runs to the next quote or whitespace.
_EMBEDDED_START: FrozenSet[str] = frozenset(" \t\n\"'=(,;")
_EMBEDDED_END: Pattern[str] = re.compile(r"[\"'\s]")
# A rule's apply reads repeated slashes and "." segments as one separator, so the prefix check does too.
_EXTRA_SEPARATORS: Pattern[str] = re.compile(r"/(?:\.?/)+")


def _spelled(path: str) -> str:
    """``path`` with forward slashes, one per separator, and no ``.`` segments, for comparing prefixes."""
    return _EXTRA_SEPARATORS.sub("/", path.replace("\\", "/"))


def _map_embedded(
    value: str, appliers: Sequence[Tuple[str, Callable[..., Tuple[bool, str]]]]
) -> str:
    """``value`` with each path embedded in it mapped by the first rule that matches that path.

    Each candidate still goes through the rule, so a rule for ``/a`` never rewrites ``/ab``.
    A source is found in either slash, because Houdini writes Windows paths with ``/``.
    """
    # Compare spelled copies, so a rule for C:\proj finds C:/proj and C://proj. The case still has to match.
    spelled: str = _spelled(value)
    keyed: List[Tuple[str, Callable[..., Tuple[bool, str]]]] = [
        (_spelled(source), apply) for source, apply in appliers if source
    ]
    if not any(source in spelled for source, _apply in keyed):
        return value
    # Walk the string, copying each character or the mapped path that starts there.
    pieces: List[str] = []
    piece: str
    i: int = 0
    while i < len(value):
        i, piece = _piece_at(value, i, keyed)
        pieces.append(piece)
    return "".join(pieces)


def _piece_at(
    value: str,
    i: int,
    keyed: Sequence[Tuple[str, Callable[..., Tuple[bool, str]]]],
) -> Tuple[int, str]:
    """The mapped path that starts at ``i`` and the index after it, else the one character there.

    ``keyed`` holds the rules' sources as ``_spelled`` writes them.
    """
    # A path can start only after a separator, and it runs to the next quote or space.
    if i > 0 and value[i - 1] in _EMBEDDED_START:
        end: Optional[re.Match[str]] = _EMBEDDED_END.search(value, i)
        j: int = end.start() if end else len(value)
        run: str = _spelled(value[i:j])
        # The first rule that maps the run wins.
        for source, apply in keyed:
            if not run.startswith(source):
                continue
            # The rule sees the whole run, so a rule for /a never maps /ab.
            changed: bool
            mapped: str
            changed, mapped = apply(path=value[i:j])
            if changed:
                return j, mapped
    return i + 1, value[i]


def _applier(rule: Any) -> Callable[..., Tuple[bool, str]]:
    """The function that applies the rule to one path, as the adaptor runtime matches paths."""
    if hasattr(rule, "apply"):
        return rule.apply
    # The client's rules are plain data. The runtime's rule has the matching logic.
    from openjd.adaptor_runtime.adaptors import PathMappingRule as RuntimeRule

    return RuntimeRule(
        source_path_format=rule.source_path_format,
        source_path=rule.source_path,
        destination_path=rule.destination_path,
        destination_os=rule.destination_os,
    ).apply


def _houdini_python(hfs: str, python_version: str, platform: str) -> str:
    """Houdini's own Python, as ``pdg.scheduler`` finds it.

    Plain Python takes no Houdini license, so a ``__PDG_PYTHON__`` command never starts a second Houdini.
    """
    if platform.startswith("win"):
        return f"{hfs}/python{python_version.replace('.', '')}/python.exe"
    if platform.startswith("darwin"):
        return (
            f"{hfs}/../../../../Python.framework/Versions/{python_version}"
            f"/bin/python{python_version}"
        )
    return f"{hfs}/python/bin/python"


def _link_scene(scene_file: str, pdg_dir: str) -> str:
    """Link the scene into ``$PDG_DIR``, where the work item's ``hip`` attribute points. Returns the scene, or ``""``."""
    if not os.path.isfile(scene_file):
        return ""
    # A link that can't be made is only a warning, because HIP still names the scene.
    link: str = os.path.join(pdg_dir, os.path.basename(scene_file))
    if not os.path.lexists(link):
        try:
            os.symlink(scene_file, link)
        except OSError as e:
            print(f"WARNING: Could not link {link} to {scene_file}: {e}", flush=True)
    return scene_file


def _stock_cook(command: str) -> Optional[Callable[[Any], Any]]:
    """The ``cook`` of the PDG job script the command starts, or None when it starts something else."""
    match: Optional[re.Match[str]] = _STOCK_COMMAND.match(command)
    if not match:
        return None
    # A script PDG doesn't ship, or one with no cook, runs through the shell instead.
    try:
        module: Any = importlib.import_module(f"pdgjob.{match.group(1)}")
    except Exception:
        return None
    cook: Any = getattr(module, "cook", None)
    return cook if callable(cook) else None


def _cook_in_process(cook: Callable[[Any], Any], pdg_dir: str) -> Tuple[bool, List[OutputFile]]:
    """Cook the staged item as ``pdg.types.cookWorkItem`` does. Returns success and the item's outputs."""
    previous: str = os.getcwd()
    os.chdir(pdg_dir)
    try:
        pdgjson: Any = importlib.import_module("pdgjob.pdgjson")
        # The first item loaded in a process stays PDG's global work item, so every
        # later `@attr` would read it. PDG's own service clients clear it first.
        importlib.import_module("pdg").EvaluationContext.setGlobalWorkItem(None, False)
        work_item: Any = pdgjson.WorkItem.fromJobEnvironment(as_native=True)
        _prepare_scene(work_item)
        # A job script's cook returns nothing, or whether it succeeded.
        result: Any = cook(work_item)
        succeeded: bool = result is None or bool(result)
        outputs: List[OutputFile] = [
            OutputFile(f.path.replace(PDG_DIR_TOKEN, pdg_dir), f.tag or "")
            for f in getattr(work_item, "outputFiles", None) or []
        ]
        return succeeded, outputs
    except SystemExit as e:
        # PDG's loader and job scripts call exit() on some failures.
        return e.code in (None, 0), []
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        return False, []
    finally:
        os.chdir(previous)


def _prepare_scene(work_item: Any) -> None:
    """Load and remap the item's ``.hip`` the way the ROP path loads its scene.

    Every item gets a freshly loaded scene, so a simulation cooked for one wedge is
    restarted for the next. The stock ``RopCooker`` is then told the scene is loaded.
    """
    # No scene, or a missing one, is left for the job script to report in its own words.
    try:
        hip: str = work_item.localizePath(work_item.fileAttribValue("hip").path).replace("\\", "/")
        modified: float = os.path.getmtime(hip)
    except Exception:
        return
    cooker: Any = _rop_cooker()
    # Imported here because the handler imports hou, which only the client has.
    try:
        from deadline.houdini_adaptor.HoudiniClient.houdini_handler import HoudiniHandler
    except ImportError:
        from houdini_adaptor.HoudiniClient.houdini_handler import (  # type: ignore[import-not-found,no-redef]
            HoudiniHandler,
        )
    # A scene that fails to load is left to the job script, which loads it itself.
    try:
        HoudiniHandler().set_scene_file({"scene_file": hip})
    except Exception as e:
        print(f"WARNING: Could not load {hip} ahead of the job script: {e}", flush=True)
        return
    if cooker is not None:
        cooker.hip_modtime = modified


def _rop_cooker() -> Any:
    """PDG's stock ``RopCooker``, or None when this Houdini has no ``pdgjob.rop``."""
    try:
        return importlib.import_module("pdgjob.rop").RopCooker
    except Exception:
        return None


def _run_shell(command: str, cwd: str) -> Tuple[int, List[str]]:
    """Run the command, echoing its output as it arrives. Returns the exit code and the ``PDG_RESULT`` lines."""
    results: List[str] = []
    process: subprocess.Popen = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    # Each line goes to the adaptor's handlers at once, and the result lines are kept.
    # A line that looks like the exit code line is prefixed, so it can't end the task early.
    assert process.stdout is not None
    for line in process.stdout:
        echoed: str = f"> {line}" if line.startswith(EXIT_CODE_MARKER) else line
        print(echoed, end="", flush=True)
        if _RESULT_PREFIX in line:
            results.append(line)
    return process.wait(), results


def _parse_pdg_results(lines: Iterable[str], pdg_dir: str) -> List[OutputFile]:
    """The files named by ``PDG_RESULT: <id>;<subindex>;<path>;<tag>;<hash>`` lines, in order."""
    outputs: List[OutputFile] = []
    for line in lines:
        if _RESULT_PREFIX not in line:
            continue
        # A line with no path names no file, so it adds nothing.
        fields: List[str] = line.split(_RESULT_PREFIX, 1)[1].strip().split(";")
        if len(fields) < 3 or not fields[2]:
            continue
        tag: str = fields[3] if len(fields) > 3 else ""
        outputs.append(OutputFile(fields[2].replace(PDG_DIR_TOKEN, pdg_dir), tag))
    return outputs


def _scan_outputs(
    output_dir: str, *, exclude: Sequence[str] = (), since: Optional[datetime] = None
) -> List[OutputFile]:
    """Every regular file under the directory, for a command that reported none.

    Skips hidden files, the excluded paths, and, with ``since``, files not
    modified since then, so the inputs synced before the task are not reported.
    """
    root: Path = Path(os.path.abspath(output_dir))
    if not root.is_dir():
        return []
    excluded: List[Path] = [Path(os.path.abspath(e)) for e in exclude if e]
    # Whole seconds, because some file systems store modification times that way.
    cutoff: Optional[int] = math.floor(since.timestamp()) if since is not None else None

    def skipped(path: Path) -> bool:
        # Hidden names, and the excluded paths with everything under them.
        return path.name.startswith(".") or any(path == e or e in path.parents for e in excluded)

    # The walk raises on a folder it can't list, so the item fails instead of recording part of it.
    found: List[Path] = []
    for folder, folders, files in os.walk(root, onerror=_raise):
        # Leave out skipped folders before the walk lists them. Folder links aren't followed.
        folders[:] = [name for name in folders if not skipped(Path(folder, name))]
        found.extend(Path(folder, name) for name in files)
    # Sorted, so the same tree always gives the same list.
    outputs: List[OutputFile] = []
    for path in sorted(found):
        # Skip hidden and excluded files, anything that isn't a regular file, and older files.
        if skipped(path) or not path.is_file():
            continue
        if cutoff is not None and path.stat().st_mtime < cutoff:
            continue
        outputs.append(OutputFile(str(path)))
    return outputs


def _raise(error: OSError) -> None:
    """Raise the error ``os.walk`` hit, so a folder the scan can't list stops the scan."""
    raise error


def _tokenize(path: str, pdg_dir: str) -> str:
    """The path with ``pdg_dir`` replaced by ``__PDG_DIR__``, only where it ends at a path separator.

    The rest of the path uses ``/`` for this OS's separator, so a cook on any OS can resolve it.
    On Windows, ``\\`` and ``/`` match each other.
    """
    root: str = pdg_dir.rstrip("/\\")
    if not root:
        return path
    # Windows takes either separator, and Houdini writes its paths with /, so compare with \ read as /.
    windows: bool = os.sep == "\\"
    seen: str = path.replace("\\", "/") if windows else path
    folder: str = root.replace("\\", "/") if windows else root
    # Match the folder only as a whole path segment, so /work/pdg never tokenizes /work/pdg2.
    if seen == folder:
        return PDG_DIR_TOKEN
    if seen.startswith(folder) and seen[len(folder)] in "/\\":
        return PDG_DIR_TOKEN + seen[len(folder) :]
    return path
