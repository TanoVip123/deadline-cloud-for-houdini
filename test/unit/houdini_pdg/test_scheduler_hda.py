# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The scheduler node's HDA sections, read as text: its parms, their defaults, and its callbacks."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Set, Tuple

import pytest

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
    assert "deadline.houdini_pdg.houdini.submit import submit" in module
    compile(module, "PythonModule", "exec")
