# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Small stand-ins for ``hou`` and ``pdg``, for the tests of the Houdini host and the registration module.

Neither module exists outside Houdini. Importing this module installs a fake
``pdg`` when there is none, and the ROP tests' fake ``hou``. Each test then sets
the scene it needs on :class:`FakeHou`, which writes onto that shared ``hou``.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest

# The ROP tests' fake ``hou``. The submitter modules bind ``hou`` when they are
# first imported, so every test in a process must share this one module.
import deadline_submitter_for_houdini.mock_hou  # type: ignore[import-not-found]  # noqa: F401


def _install_pdg() -> types.ModuleType:
    """The fake ``pdg`` module, with the enums and base class the scheduler code names."""
    # Reuse a fake another test module installed, so every test sees one module.
    existing: Optional[types.ModuleType] = sys.modules.get("pdg")
    if existing is not None and hasattr(existing, "scheduleResult"):
        return existing
    module: types.ModuleType = types.ModuleType("pdg")
    module.scheduleResult = types.SimpleNamespace(  # type: ignore[attr-defined]
        Succeeded="Succeeded", Deferred="Deferred", CookFailed="CookFailed"
    )
    module.tickResult = types.SimpleNamespace(  # type: ignore[attr-defined]
        SchedulerReady="SchedulerReady", SchedulerCancelCook="SchedulerCancelCook"
    )
    module.EventType = types.SimpleNamespace(  # type: ignore[attr-defined]
        NodeGenerated="NodeGenerated"
    )
    scheduler: types.ModuleType = types.ModuleType("pdg.scheduler")
    scheduler.PyScheduler = FakePyScheduler  # type: ignore[attr-defined]
    module.scheduler = scheduler  # type: ignore[attr-defined]
    sys.modules["pdg"] = module
    sys.modules["pdg.scheduler"] = scheduler
    return module


class FakePyScheduler:
    """PDG's ``PyScheduler`` base: what the scheduler forwards to PDG, recorded in ``calls``."""

    def __init__(self, scheduler: Any = None, name: str = "deadlinecloudscheduler1") -> None:
        self.name: str = name
        self.calls: List[Tuple[Any, ...]] = []
        self.context: Any = None
        self.node: Any = None
        self.dependency_graph: Any = ({}, {}, [])

    def topNode(self) -> Any:
        return self.node

    def dependencyGraph(self, expand: bool) -> Any:
        self.calls.append(("dependencyGraph", expand))
        return self.dependency_graph

    def setWorkingDir(self, local: str, remote: str) -> None:
        self.calls.append(("setWorkingDir", local, remote))

    def cookError(self, message: str) -> None:
        self.calls.append(("cookError", message))

    def onWorkItemStartCook(self, *args: Any) -> None:
        self.calls.append(("onWorkItemStartCook",) + args)

    def onWorkItemAddOutput(self, *args: Any) -> None:
        self.calls.append(("onWorkItemAddOutput",) + args)

    def onWorkItemSucceeded(self, *args: Any) -> None:
        self.calls.append(("onWorkItemSucceeded",) + args)

    def onWorkItemFailed(self, *args: Any) -> None:
        self.calls.append(("onWorkItemFailed",) + args)


PDG: types.ModuleType = _install_pdg()


# -- HOM ----------------------------------------------------------------------


class _Named:
    """Anything HOM exposes through ``name()``, such as a category or a parm type."""

    def __init__(self, name: str) -> None:
        self._name: str = name

    def name(self) -> str:
        return self._name


class _Template:
    """A ``hou.ParmTemplate``: String with a string type, or another type."""

    def __init__(self, kind: str) -> None:
        self._kind: str = kind

    def type(self) -> _Named:
        return _Named("String" if self._kind else "Int")

    def stringType(self) -> _Named:
        return _Named(self._kind)

    def tags(self) -> Dict[str, str]:
        return {}


class FakeParm:
    """A ``hou.Parm``. ``kind`` is its string type, such as ``FileReference``, or ``""`` for a number."""

    def __init__(self, node: "FakeNode", name: str, value: Any, kind: str = "") -> None:
        self._node: FakeNode = node
        self._name: str = name
        self.value: Any = value
        self.kind: str = kind
        self.sets: List[Any] = []

    def name(self) -> str:
        return self._name

    def path(self) -> str:
        return f"{self._node.path()}/{self._name}"

    def node(self) -> "FakeNode":
        return self._node

    def parmTemplate(self) -> _Template:
        return _Template(self.kind)

    def eval(self) -> Any:
        # A parm set to an exception stands for one whose expression fails.
        if isinstance(self.value, Exception):
            raise self.value
        return expand(str(self.value)) if self.kind else self.value

    def evalAsString(self) -> str:
        return expand(str(self.eval()))

    def unexpandedString(self) -> str:
        return str(self.value)

    def expression(self) -> str:
        raise RuntimeError("no channel expression")

    def set(self, value: Any) -> None:
        self.sets.append(value)
        self.value = value


class FakeNode:
    """A ``hou.Node`` with only what the host, the readers, and the ROP's scan call."""

    def __init__(
        self,
        path: str,
        *,
        parms: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, str]] = None,
        category: str = "Top",
        type_name: str = "node",
    ) -> None:
        self._path: str = path
        self._parent: Optional[FakeNode] = None
        self._children: List[FakeNode] = []
        self._inputs: List[FakeNode] = []
        self._category: str = category
        self._type_name: str = type_name
        self._parms: Dict[str, FakeParm] = {
            name: FakeParm(self, name, value) for name, value in (parms or {}).items()
        }
        # File-reference parms, which the scene walk and the ROP's scan read.
        for name, value in (files or {}).items():
            self._parms[name] = FakeParm(self, name, value, kind="FileReference")
        self.pdg_node: Any = None
        self.context: Any = None
        self.child_category: str = "Top"
        self.cook_calls: List[Dict[str, Any]] = []
        self.spare: List[FakeParm] = []
        self.display: Optional[FakeNode] = None

    def path(self) -> str:
        return self._path

    def name(self) -> str:
        return self._path.rsplit("/", 1)[-1]

    def parent(self) -> Optional["FakeNode"]:
        return self._parent

    def add(self, child: "FakeNode", *inputs: "FakeNode") -> "FakeNode":
        # Parent and wire ``child`` the way Houdini does.
        child._parent = self
        child._inputs = list(inputs)
        self._children.append(child)
        return child

    def children(self) -> List["FakeNode"]:
        return list(self._children)

    def allSubChildren(self) -> List["FakeNode"]:
        return list(self._children)

    def inputs(self) -> List["FakeNode"]:
        return list(self._inputs)

    def inputAncestors(self) -> List["FakeNode"]:
        return []

    def parms(self) -> List[FakeParm]:
        return list(self._parms.values())

    def parm(self, name: str) -> Optional[FakeParm]:
        return self._parms.get(name)

    def set_parm(self, name: str, value: Any, kind: str = "") -> FakeParm:
        """Add or replace a parm, and return it."""
        parm: FakeParm = FakeParm(self, name, value, kind)
        self._parms[name] = parm
        return parm

    def spareParms(self) -> List[FakeParm]:
        return list(self.spare)

    def childTypeCategory(self) -> _Named:
        return _Named(self.child_category)

    def type(self) -> Any:
        return types.SimpleNamespace(
            category=lambda: _Named(self._category), name=lambda: self._type_name
        )

    def node(self, ref: str) -> Any:
        # A child by name, or any child by its absolute path.
        return next((c for c in self._children if ref in (c.name(), c.path())), None)

    def displayNode(self) -> Optional["FakeNode"]:
        return self.display

    def isInsideLockedHDA(self) -> bool:
        return False

    def getPDGNode(self) -> Any:
        return self.pdg_node

    def getPDGNodeName(self) -> Optional[str]:
        return None

    def getPDGGraphContext(self) -> Any:
        return self.context

    def cookWorkItems(self, **kwargs: Any) -> None:
        self.cook_calls.append(kwargs)


class FakeHipFile:
    """``hou.hipFile``: the scene's path, whether it is new or unsaved, and how often it was saved."""

    def __init__(self, path: str) -> None:
        self._path: str = path
        self.new: bool = False
        self.unsaved: bool = False
        self.saves: int = 0

    def path(self) -> str:
        return self._path

    def isNewFile(self) -> bool:
        return self.new

    def hasUnsavedChanges(self) -> bool:
        return self.unsaved

    def save(self) -> None:
        self.saves += 1
        self.unsaved = False


#: The scene folder ``$HIP`` expands to, set by :class:`FakeHou`.
_HIP_DIR: Dict[str, str] = {"value": os.sep}


def expand(text: str) -> str:
    """Houdini's expansion, for the variables the tests use: ``$HIP``, ``$F4``, and the environment."""
    # Frame 1, as a scene at its first frame evaluates ``$F4``.
    text = text.replace("${HIP}", _HIP_DIR["value"]).replace("$HIP", _HIP_DIR["value"])
    return os.path.expandvars(text.replace("$F4", "0001"))


class FakeMessages:
    """``hou.ui.displayMessage``: answers from ``answers`` in turn, then ``choice``, and records each call's buttons."""

    def __init__(self) -> None:
        self.choice: int = 0
        self.answers: List[int] = []
        self.calls: List[Tuple[str, Tuple[str, ...]]] = []

    def __call__(self, text: str, **kwargs: Any) -> int:
        self.calls.append((text, tuple(kwargs.get("buttons", ()))))
        return self.answers.pop(0) if self.answers else self.choice


@dataclass
class FakeHou:
    """The scene a test sets up, written onto the shared fake ``hou`` module."""

    hip_file: FakeHipFile
    messages: FakeMessages
    ui: bool = True
    file_references: List[Tuple[Any, str]] = field(default_factory=list)
    instances: List[Any] = field(default_factory=list)
    libraries: Dict[str, str] = field(default_factory=dict)
    version: str = "21.0.596"
    #: Raised by ``hou.fileReferences`` when set, as a scene that cannot be scanned.
    scan_error: Optional[Exception] = None


def _file_references(state: FakeHou) -> List[Tuple[Any, str]]:
    """The scene's file references, or the scan error a test set."""
    if state.scan_error is not None:
        raise state.scan_error
    return list(state.file_references)


def install_hou(monkeypatch: pytest.MonkeyPatch, hip_file: str) -> FakeHou:
    """Point the shared fake ``hou`` at a scene saved as ``hip_file``, for one test."""
    hou: Any = sys.modules["hou"]
    _HIP_DIR["value"] = os.path.dirname(hip_file)
    state: FakeHou = FakeHou(hip_file=FakeHipFile(hip_file), messages=FakeMessages())

    def node_type(category: str, name: str) -> Any:
        # A node type's instances, and the library that defines it, when a test set one.
        path: Optional[str] = state.libraries.get(name)
        definition: Any = (
            types.SimpleNamespace(libraryFilePath=lambda: path) if path is not None else None
        )
        return types.SimpleNamespace(
            instances=lambda: tuple(state.instances), definition=lambda: definition
        )

    # Each attribute reads the state, so a test can change it after this call.
    attributes: Dict[str, Any] = {
        "hipFile": state.hip_file,
        "text": types.SimpleNamespace(expandString=expand),
        "ui": types.SimpleNamespace(displayMessage=state.messages),
        "isUIAvailable": lambda: state.ui,
        "fileReferences": lambda: _file_references(state),
        "getenv": lambda name: os.environ.get(name),
        "nodeType": node_type,
        "topNodeTypeCategory": lambda: "Top",
        "ropNodeTypeCategory": lambda: "Driver",
        "applicationVersionString": lambda: state.version,
        "ToggleParmTemplate": type("ToggleParmTemplate", (), {}),
        "severityType": types.SimpleNamespace(Error="Error", Message="Message"),
    }
    for name, value in attributes.items():
        monkeypatch.setattr(hou, name, value, raising=False)
    return state


# -- PDG ----------------------------------------------------------------------


@dataclass
class FakePdgNode:
    """A ``pdg.Node``: its job script, type, scheduler, and its TOP node."""

    name: str
    command: str = ""
    node_type: str = "Processor"
    isDynamic: bool = False
    scheduler_name: str = "deadlinecloudscheduler1"
    hom: Any = None
    workItems: List[Any] = field(default_factory=list)
    #: None stands for the Automatic every node reports before generation.
    cookType: Any = None

    @property
    def scriptInfo(self) -> Any:
        return types.SimpleNamespace(command=self.command)

    @property
    def nodeType(self) -> Any:
        return types.SimpleNamespace(name=self.node_type)

    @property
    def scheduler(self) -> Any:
        return types.SimpleNamespace(name=self.scheduler_name)

    def topNode(self) -> Any:
        return self.hom


@dataclass
class FakeContext:
    """A PDG graph context: its nodes, the cook set, and its event handlers."""

    nodes: List[FakePdgNode]
    cookSet: Any = None
    handlers: List[Tuple[Any, Any]] = field(default_factory=list)
    items: Dict[int, Any] = field(default_factory=dict)

    @property
    def graph(self) -> Any:
        return types.SimpleNamespace(
            nodes=lambda: list(self.nodes), workItemById=lambda item_id: self.items.get(item_id)
        )

    def addEventHandler(self, function: Any, event_type: Any, is_python: bool) -> Any:
        handler: Tuple[Any, Any] = (function, event_type)
        self.handlers.append(handler)
        return handler

    def removeEventHandler(self, handler: Any) -> None:
        self.handlers.remove(handler)

    def emit(self, event_type: Any, node_name: str) -> None:
        """Send ``event_type`` for the named node to every handler registered for it."""
        node: FakePdgNode = next(n for n in self.nodes if n.name == node_name)
        for function, kind in list(self.handlers):
            if kind == event_type:
                function(None, types.SimpleNamespace(type=kind, node=node))


SCHEDULER_PATH: str = "/obj/topnet1/deadlinecloudscheduler1"


@dataclass
class Network:
    """A TOP network, its scheduler node, and PDG's view of both."""

    topnet: FakeNode
    scheduler_node: FakeNode
    pdg_nodes: List[FakePdgNode]
    context: FakeContext

    def scheduler(self) -> FakePyScheduler:
        """The PDG scheduler object for the network's scheduler node."""
        scheduler: FakePyScheduler = FakePyScheduler()
        scheduler.context = self.context
        scheduler.node = self.scheduler_node
        return scheduler

    def node(self, name: str) -> FakePdgNode:
        """The PDG node called ``name``."""
        return next(n for n in self.pdg_nodes if n.name == name)


def build_network(
    spec: Sequence[Tuple[Any, ...]],
    *,
    scheduler_parms: Optional[Dict[str, Any]] = None,
) -> Network:
    """``spec`` rows are (name, command, PDG node type, input names[, cook type])."""
    obj: FakeNode = FakeNode("/obj", category="Object")
    obj.child_category = "Object"
    topnet: FakeNode = obj.add(FakeNode("/obj/topnet1", category="Object"))
    homs: Dict[str, FakeNode] = {}
    pdg_nodes: List[FakePdgNode] = []
    # One TOP node and one PDG node per row, wired to the rows it names.
    for row in spec:
        name, command, node_type, inputs = row[:4]
        cook_type: Any = row[4] if len(row) > 4 else None
        hom: FakeNode = topnet.add(FakeNode(f"/obj/topnet1/{name}"), *[homs[i] for i in inputs])
        node: FakePdgNode = FakePdgNode(
            name=name, command=command, node_type=node_type, hom=hom, cookType=cook_type
        )
        hom.pdg_node = node
        homs[name] = hom
        pdg_nodes.append(node)
    parms: Dict[str, Any] = (
        {"working_dir": "$HIP", "debug_level": 0, "validate_graph": 1}
        if scheduler_parms is None
        else scheduler_parms
    )
    scheduler_node: FakeNode = topnet.add(FakeNode(SCHEDULER_PATH, parms=parms))
    # The network cooks on this scheduler by default, and its last node is displayed.
    topnet.set_parm("topscheduler", "deadlinecloudscheduler1")
    topnet.display = homs[spec[-1][0]] if spec else None
    context: FakeContext = FakeContext(nodes=pdg_nodes)
    scheduler_node.context = context
    return Network(topnet, scheduler_node, pdg_nodes, context)


#: cooking_example's shape: an in-process Python Script between two steps.
COOKING_EXAMPLE: List[Tuple[Any, ...]] = [
    ("subdivide", "hython subdivide.py", "Processor", []),
    ("python1", "", "Processor", ["subdivide"], "InProcess"),
    ("render", "hython render.py", "Processor", ["python1"]),
]

#: The same shape, routed through a Merge. A Merge cooks nothing, so it is accepted.
ROUTED_EXAMPLE: List[Tuple[Any, ...]] = [
    ("subdivide", "hython subdivide.py", "Processor", []),
    ("merge", "", "Processor", ["subdivide"], "Generate"),
    ("render", "hython render.py", "Processor", ["merge"]),
]

#: Pyro Bonfire's shape: a wedge root, then steps, joined by a partitioner.
PYRO_BONFIRE: List[Tuple[Any, ...]] = [
    ("wedge1", "", "Processor", []),
    ("sim", "hython sim.py", "Processor", ["wedge1"]),
    ("render", "hython render.py", "Processor", ["sim"]),
    ("partition", "", "Partitioner", ["render"]),
    ("montage", "magick montage", "Processor", ["partition"]),
]


def main_thread_runner(ran_on: List[str]) -> Callable[..., Any]:
    """An ``hdefereval.executeInMainThreadWithResult`` that records the thread it was called from."""
    import threading

    def execute(function: Callable[..., Any], *args: Any) -> Any:
        ran_on.append(threading.current_thread().name)
        return function(*args)

    return execute
