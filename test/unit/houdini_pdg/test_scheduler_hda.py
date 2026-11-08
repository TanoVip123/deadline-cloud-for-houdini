# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The scheduler node's HDA sections, read as text: its parms, their defaults, and its callbacks."""

from __future__ import annotations

import logging
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest

from .houdini_fakes import FakeHou, FakeNode, install_hou

_SRC: str = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "deadline")
_NODE: str = os.path.join(
    _SRC,
    "houdini_pdg",
    "houdini_plugin",
    "otls",
    "deadline_cloud_scheduler.hda",
    "Top_1deadlinecloudscheduler",
)
_ROP: str = os.path.join(
    _SRC, "houdini_submitter", "otls", "deadline_cloud.hda", "Driver_1deadline__cloud"
)


def _read(folder: str, section: str) -> str:
    """One section of an expanded HDA."""
    with open(os.path.join(folder, section), encoding="utf8") as f:
        return f.read()


def _parms(dialog_script: str) -> Dict[str, str]:
    """Each parm's name, with the text of its block."""
    blocks: Dict[str, str] = {}
    # A parm block starts at ``parm {`` and its name is the next ``name`` line.
    for match in re.finditer(r"\bparm\s*\{\s*name\s+\"?(\w+)\"?(.*?)\n\s*\}", dialog_script, re.S):
        blocks[match.group(1)] = match.group(2)
    return blocks


NODE_PARMS: Dict[str, str] = _parms(_read(_NODE, "DialogScript"))


@pytest.mark.parametrize(
    "parm, default",
    [
        ("working_dir", "$HIP"),
        ("debug_level", "0"),
        ("validate_graph", "1"),
        ("auto_save_hip", "0"),
        ("max_worker_count", "0"),
    ],
)
def test_the_node_declares_each_parm_the_scheduler_reads_with_its_default(
    parm: str, default: str
) -> None:
    """A missing parm falls back to a default silently, so each one the cook reads must exist."""
    assert parm in NODE_PARMS
    assert re.search(r'default\s*\{\s*"' + re.escape(default) + r'"\s*\}', NODE_PARMS[parm])


def test_the_debug_level_offers_two_levels() -> None:
    """Higher levels add nothing, so offering them would mislead the artist."""
    menu: List[Tuple[str, str]] = re.findall(r'"(\d)"\s+"([^"]+)"', NODE_PARMS["debug_level"])
    assert menu == [("0", "Standard"), ("1", "Graph dump and item details")]


def test_every_button_calls_the_nodes_callback() -> None:
    """A button wired elsewhere would run the ROP's flow, such as its submit dialog."""
    buttons: Set[str] = {
        name for name, block in NODE_PARMS.items() if re.search(r"type\s+button", block)
    }
    assert buttons == {"login", "logout", "settings", "update_queue_parameters", "submit"}
    for name in buttons:
        assert '"script_callback" "hou.phm().callback(kwargs)"' in NODE_PARMS[name], name


def test_the_node_has_the_rops_shared_job_settings() -> None:
    """The job settings work the same way on both nodes, so they share names with the ROP."""
    rop: Dict[str, str] = _parms(_read(_ROP, "DialogScript"))
    shared: Set[str] = {
        "name",
        "description",
        "priority",
        "failed_tasks_limit",
        "task_retry_limit",
        "farm",
        "queue",
        "login",
        "logout",
        "settings",
        "include_adaptor_wheels",
        "adaptor_wheels",
        "auto_save_hip",
        "submit",
    }
    assert shared <= set(rop) and shared <= set(NODE_PARMS)


def test_every_section_the_node_lists_exists() -> None:
    """Houdini loads only the sections the list names, so a missing one would drop the callbacks."""
    listed: List[str] = [
        line.split()[0] for line in _read(_NODE, "Sections.list").splitlines()[1:] if line.strip()
    ]
    assert {"PythonModule", "OnCreated", "OnLoaded"} <= set(listed)
    for section in listed:
        assert os.path.isfile(os.path.join(_NODE, section)), section


def test_the_python_module_sends_submit_to_the_schedulers_own_flow() -> None:
    """Submit must not open the ROP's dialog, so the module names the scheduler's submit."""
    module: str = _read(_NODE, "PythonModule")
    assert "deadline.houdini_pdg.houdini.host import submit" in module
    compile(module, "PythonModule", "exec")


#: The two places the node's module looks for the ROP's submitter, in order.
_ROP_INSTALL: str = "deadline_cloud_for_houdini.submitter"
_PACKAGE_COPY: str = "deadline.houdini_submitter.python.deadline_cloud_for_houdini.submitter"


def _python_module() -> types.ModuleType:
    """The node's PythonModule section, run as Houdini runs it for ``hou.phm()``."""
    module: types.ModuleType = types.ModuleType("deadlinecloudscheduler_phm")
    exec(compile(_read(_NODE, "PythonModule"), "PythonModule", "exec"), module.__dict__)
    return module


class _SchedulerNode(FakeNode):
    """A scheduler node whose ``hdaModule`` is the node's PythonModule."""

    def __init__(self, phm: types.ModuleType) -> None:
        super().__init__("/obj/topnet1/deadlinecloudscheduler1")
        self.phm: types.ModuleType = phm

    def hdaModule(self) -> types.ModuleType:
        return self.phm


def _without(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make each module in ``names`` fail to import, as on a machine without it."""
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


@pytest.mark.parametrize("section", ["OnCreated", "OnLoaded"])
def test_the_node_is_created_and_loaded_without_the_rop_submitter(
    monkeypatch: pytest.MonkeyPatch, caplog: Any, section: str
) -> None:
    """A missing submitter used to raise from OnCreated, which shows a traceback for every new node."""
    # Run the section as Houdini does, with neither copy of the submitter importable.
    _without(monkeypatch, _ROP_INSTALL, _PACKAGE_COPY)
    node: _SchedulerNode = _SchedulerNode(_python_module())
    with caplog.at_level(logging.WARNING):
        exec(compile(_read(_NODE, section), section, "exec"), {"kwargs": {"node": node}})
    assert "submitter for Houdini isn't installed" in caplog.text
    assert node.path() in caplog.text


def test_without_the_rops_install_the_node_uses_this_packages_submitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler ships the ROP's module too, so the node works where only it is installed."""
    _without(monkeypatch, _ROP_INSTALL)
    found: Any = _python_module()._rop_submitter()
    assert found is not None and found.__name__ == _PACKAGE_COPY


def test_a_rop_button_without_the_submitter_says_what_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Login and Settings buttons can't work without the submitter, so the artist is told why."""
    # Press Login with no submitter, and expect one message that names it.
    scene: FakeHou = install_hou(monkeypatch, str(tmp_path / "shot.hip"))
    _without(monkeypatch, _ROP_INSTALL, _PACKAGE_COPY)
    phm: types.ModuleType = _python_module()
    parm: Any = types.SimpleNamespace(name=lambda: "login")
    phm.callback({"parm": parm, "node": _SchedulerNode(phm)})
    ((text, _),) = scene.messages.calls
    assert "submitter for Houdini" in text
