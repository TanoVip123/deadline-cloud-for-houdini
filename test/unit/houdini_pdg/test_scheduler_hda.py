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
    assert "deadline.houdini_pdg.houdini.host import submit" in module
    compile(module, "PythonModule", "exec")


@pytest.mark.parametrize(
    "parm", ["priority", "failed_tasks_limit", "task_retry_limit", "max_worker_count"]
)
def test_the_job_number_parms_are_plain_integer_fields(parm: str) -> None:
    """The artist types these counts, and a slider made them hard to set exactly."""
    block: str = NODE_PARMS[parm]
    assert re.search(r"type\s+integer", block)
    assert re.search(r'parmtag\s*\{\s*"sidefx::slider"\s+"none"\s*\}', block), parm


def _groups(dialog_script: str) -> List[Tuple[str, str]]:
    """Each top-level tab, as (name, label), in the order the node shows them."""
    return re.findall(r'\n    group \{\s*name\s+"(\w+)"\s*label\s+"([^"]+)"', dialog_script)


def _lists(dialog_script: str) -> Dict[str, Tuple[str, str]]:
    """Each multiparm or multiscroll, with its kind and the text of its block."""
    return {
        match.group(2): (match.group(1), match.group(3))
        for match in re.finditer(
            r"\b(multiparm|multiscroll)\s*\{\s*name\s+\"(\w+)\"(.*?)\n        \}",
            dialog_script,
            re.S,
        )
    }


NODE_SCRIPT: str = _read(_NODE, "DialogScript")


def test_the_node_has_the_job_attachments_and_host_requirements_tabs_after_pdg() -> None:
    """The artist asked for the ROP's two tabs, so they follow the node's own tabs."""
    assert _groups(NODE_SCRIPT) == [
        ("job_settings", "Shared Job Settings"),
        ("job_settings_1", "PDG"),
        ("job_settings_2", "Job Attachments"),
        ("host_requirements", "Host Requirements"),
    ]


@pytest.mark.parametrize(
    "name, entry, kind",
    [
        ("input_filenames", "input_filenames#", "file"),
        ("input_directories", "input_dir#", "directory"),
        ("output_directories", "output_dir#", "directory"),
    ],
)
def test_the_attachment_lists_match_the_rops(name: str, entry: str, kind: str) -> None:
    """The ROP's reader evaluates these by name, so a renamed parm would read as an empty list."""
    for script in (NODE_SCRIPT, _read(_ROP, "DialogScript")):
        list_kind, block = _lists(script)[name]
        assert list_kind == "multiscroll"
        assert re.search(r'name\s+"' + re.escape(entry) + r'"\s.*?type\s+' + kind, block, re.S)


def test_the_node_has_no_parse_files_button_and_no_stored_scan() -> None:
    """The cook scans the scene every time, so a stored scan would only go stale."""
    names: Set[str] = set(_lists(NODE_SCRIPT)) | set(NODE_PARMS)
    assert "parse_files" not in names
    assert not {name for name in names if name.startswith("auto_")} - {"auto_save_hip"}


_HOST_PARMS: List[Tuple[str, str]] = [("hr_os", '""'), ("hr_cpu_arch", '""')] + [
    (f"hr_{field}_{end}", '"-1"')
    for field in ("vcpu", "memory", "gpu", "gpu_memory", "scratch")
    for end in ("min", "max")
]


@pytest.mark.parametrize("parm, default", _HOST_PARMS)
def test_each_host_requirement_defaults_to_not_set_and_waits_for_custom(
    parm: str, default: str
) -> None:
    """The host reads -1 as not set, and a field that works with "all workers" chosen would mislead."""
    block: str = NODE_PARMS[parm]
    assert re.search(r"default\s*\{\s*" + re.escape(default) + r"\s*\}", block), parm
    assert 'disablewhen "{ hr_mode == all }"' in block, parm


def test_the_host_requirements_mode_offers_all_workers_first() -> None:
    """A new node must run on any worker, as a ROP whose tab was never opened does."""
    block: str = NODE_PARMS["hr_mode"]
    assert re.findall(r'"(\w+)"\s+"[^"]+"', block) == ["all", "custom"]
    assert re.search(r'default\s*\{\s*"0"\s*\}', block)


@pytest.mark.parametrize(
    "name, entries",
    [
        ("hr_amounts", ["hr_amount_name#", "hr_amount_min#", "hr_amount_max#"]),
        ("hr_attributes", ["hr_attr_name#", "hr_attr_option#", "hr_attr_values#"]),
    ],
)
def test_the_custom_requirement_lists_have_the_entries_the_host_reads(
    name: str, entries: List[str]
) -> None:
    """The host reads each entry by name and number, so a renamed one would be dropped."""
    kind, block = _lists(NODE_SCRIPT)[name]
    assert kind == "multiparm" and 'disablewhen "{ hr_mode == all }"' in block
    assert re.findall(r'name\s+"(hr_\w+#)"', block) == entries
