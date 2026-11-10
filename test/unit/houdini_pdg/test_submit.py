# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The scheduler node's Submit button, against small fakes of ``hou`` and ``pdg``."""

from __future__ import annotations

import os
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from .houdini_fakes import (
    PYRO_BONFIRE,
    COOKING_EXAMPLE,
    FakeHou,
    Network,
    build_network,
    install_hou,
)

from deadline.houdini_pdg.houdini.host import submit

_SAVE: Tuple[str, ...] = ("Save", "Always Save", "Cancel")
_CONFIRM: Tuple[str, ...] = ("Submit", "Cancel")


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeHou:
    """A saved scene at ``<tmp>/scene/shot.hip``, with a UI."""
    hip: str = os.path.join(str(tmp_path), "scene", "shot.hip")
    os.makedirs(os.path.dirname(hip))
    monkeypatch.delenv("DEADLINE_PDG_DEBUG_LEVEL", raising=False)
    monkeypatch.setenv("HIP", os.path.dirname(hip))
    return install_hou(monkeypatch, hip)


def _workstation(name: str) -> str:
    """A workstation with a default farm and queue."""
    values: Dict[str, str] = {"defaults.farm_id": "farm-1", "defaults.queue_id": "queue-1"}
    return values.get(name, "")


def _network() -> Network:
    """A Pyro Bonfire network whose scheduler node has every parm Submit reads."""
    return build_network(
        PYRO_BONFIRE,
        scheduler_parms={
            "working_dir": "$HIP",
            "debug_level": 0,
            "validate_graph": 1,
            "auto_save_hip": 0,
            "name": "bonfire",
            "priority": 50,
        },
    )


def _cooks(network: Network) -> List[Dict[str, Any]]:
    """The cooks the display node was asked for."""
    display: Any = network.topnet.displayNode()
    return list(display.cook_calls)


def _parm_writes(network: Network) -> List[Any]:
    """Every value written to any scheduler node parm."""
    return [value for parm in network.scheduler_node.parms() for value in parm.sets]


def test_a_saved_scene_submits_after_one_confirmation_and_stays_saved(scene: FakeHou) -> None:
    """A saved scene gets no save prompt, and Submit writes no parm, so the scene stays saved."""
    network: Network = _network()
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    assert [buttons for _, buttons in scene.messages.calls] == [_CONFIRM]
    assert scene.hip_file.hasUnsavedChanges() is False and scene.hip_file.saves == 0
    assert _parm_writes(network) == []
    assert _cooks(network) == [{"block": False}]


def test_an_unsaved_scene_shows_the_save_prompt_before_any_other_dialog(scene: FakeHou) -> None:
    """The artist answers the save question first, and the confirmation follows the save."""
    scene.hip_file.unsaved = True
    network: Network = _network()
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    assert [buttons for _, buttons in scene.messages.calls] == [_SAVE, _CONFIRM]
    assert scene.hip_file.saves == 1


@pytest.mark.parametrize("answers", [[2], [0, 1]])
def test_cancel_cooks_nothing_and_writes_nothing(scene: FakeHou, answers: List[int]) -> None:
    """Cancel at the save prompt or at the confirmation leaves the scene and the node as they were."""
    scene.hip_file.unsaved = True
    scene.messages.answers = list(answers)
    network: Network = _network()
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert _cooks(network) == []
    assert _parm_writes(network) == []


def test_a_network_that_cooks_on_another_scheduler_is_refused(scene: FakeHou) -> None:
    """Cooking the display node uses the network's default scheduler, which could be another one."""
    network: Network = _network()
    network.topnet.set_parm("topscheduler", "localscheduler")
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert _cooks(network) == []
    assert "Default TOP Scheduler" in scene.messages.calls[0][0]


def test_enter_picks_save_on_the_save_prompt_and_submit_on_the_confirmation(
    scene: FakeHou,
) -> None:
    """Enter must never pick Always Save, which writes a parm, or Cancel instead of Submit."""
    # Submit an unsaved scene, then name the button Enter and the close box pick on each dialog.
    scene.hip_file.unsaved = True
    network: Network = _network()
    submit(network.scheduler_node, get_setting=_workstation)
    picked: List[Tuple[str, str]] = [
        (buttons[options["default_choice"]], buttons[options["close_choice"]])
        for (_, buttons), options in zip(scene.messages.calls, scene.messages.options)
    ]
    assert picked == [("Save", "Cancel"), ("Submit", "Cancel")]


def test_a_parent_network_without_a_default_scheduler_parm_is_refused(scene: FakeHou) -> None:
    """A scheduler inside a TOP subnet has a parent with no such parm, and Submit must say so, not crash."""
    # A parent with no Default TOP Scheduler parm gets the message and no cook.
    network: Network = _network()
    network.topnet.remove_parm("topscheduler")
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert _cooks(network) == []
    assert "Default TOP Scheduler" in scene.messages.calls[0][0]


def test_a_refused_graph_is_named_and_not_cooked(scene: FakeHou) -> None:
    """The artist sees why before anything is uploaded, and no confirmation is offered."""
    network: Network = build_network(
        COOKING_EXAMPLE, scheduler_parms={"working_dir": "$HIP", "validate_graph": 1}
    )
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert _cooks(network) == []
    ((text, buttons),) = scene.messages.calls
    assert "python1" in text and buttons == ()
    assert scene.messages.options[0]["severity"] == "Error"


@pytest.mark.parametrize(
    "settings, words",
    [({}, "farm ID"), ({"defaults.farm_id": "farm-1"}, "queue ID")],
)
def test_a_workstation_without_a_farm_or_queue_is_told_which_setting_is_missing(
    scene: FakeHou, settings: Dict[str, str], words: str
) -> None:
    """With no farm or queue the job has nowhere to go, so the ROP's message names the setting."""
    network: Network = _network()
    assert submit(network.scheduler_node, get_setting=lambda name: settings.get(name, "")) is False
    assert words in scene.messages.calls[0][0]
    assert scene.messages.options[0]["severity"] == "Error"
    assert _cooks(network) == []


def test_the_workstation_defaults_are_read_when_no_reader_is_given(
    scene: FakeHou, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The node's Submit reads the same farm and queue as the ROP, from the workstation config."""
    from configparser import ConfigParser

    from deadline.client.config import config_file

    # A config file with a farm and a queue, which the confirmation then shows.
    parser: ConfigParser = ConfigParser()
    config_file.set_setting("defaults.farm_id", "farm-cfg", config=parser)
    config_file.set_setting("defaults.queue_id", "queue-cfg", config=parser)
    path: Path = tmp_path / "config"
    with open(path, "w", encoding="utf8") as f:
        parser.write(f)
    monkeypatch.setenv("DEADLINE_CONFIG_FILE_PATH", str(path))
    assert submit(_network().scheduler_node) is True
    assert "Farm: farm-cfg" in scene.messages.calls[0][0]


def test_the_confirmation_shows_the_jobs_settings_and_any_warning(scene: FakeHou) -> None:
    """The artist confirms what will run, and a waived refusal is shown, not hidden."""
    network: Network = build_network(
        COOKING_EXAMPLE,
        scheduler_parms={"working_dir": "$HIP", "validate_graph": 0, "priority": 70},
    )
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    text: str = scene.messages.calls[0][0]
    # With no name parm the job takes the scene's file name.
    assert "Job name: shot.hip" in text and "Priority: 70" in text
    assert "Farm: farm-1" in text and "Queue: queue-1" in text
    assert "python1" in scene.messages.options[0]["details"]


def test_closing_the_confirmation_cooks_nothing(scene: FakeHou) -> None:
    """Closing the window is a cancel, so a stray click never submits a job."""
    scene.messages.answers = [None]
    network: Network = _network()
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert _cooks(network) == []


def test_a_wrong_working_directory_is_named_before_any_confirmation(scene: FakeHou) -> None:
    """The cook would fail on it at once, so Submit says which parm to fix and cooks nothing."""
    network: Network = build_network(PYRO_BONFIRE, scheduler_parms={"working_dir": ""})
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    ((text, buttons),) = scene.messages.calls
    assert "'working_dir'" in text and buttons == ()
    assert scene.messages.options[0]["severity"] == "Error"
    assert _cooks(network) == []


def test_the_refusal_popup_lists_one_refusal_per_line(scene: FakeHou) -> None:
    """The popup uses the node error's text, a header and then one line per refusal."""
    import re

    network: Network = build_network(
        COOKING_EXAMPLE, scheduler_parms={"working_dir": "$HIP", "validate_graph": 1}
    )
    submit(network.scheduler_node, get_setting=_workstation)
    lines: List[str] = scene.messages.calls[0][0].split("\n")
    header: int = next(
        i for i, line in enumerate(lines) if line.startswith("The graph was refused")
    )
    count: int = int(re.search(r"\((\d+) refusal", lines[header]).group(1))  # type: ignore[union-attr]
    assert count >= 1 and len(lines) == header + 1 + count


def test_submitting_a_display_node_with_no_step_above_it_is_refused(scene: FakeHou) -> None:
    """With the wedge displayed, the cook would run only in-process nodes and make no job."""
    # Display the wedge, then Submit, which must refuse before cooking.
    network: Network = _network()
    network.topnet.display = network.node("wedge1").hom
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert _cooks(network) == []
    assert "No node in this cook runs a command on a worker" in scene.messages.calls[0][0]


class _Item:
    """A generated ``pdg.WorkItem`` with a command, so its node becomes a step."""

    def __init__(self, node: str, index: int, command: str) -> None:
        # The fields the graph read takes from a work item.
        self.name: str = f"{node}_{index}"
        self.index: int = index
        self.id: int = 1000 + index
        self.node: Any = types.SimpleNamespace(name=node)
        self.command: str = command
        self.state: Any = types.SimpleNamespace(name="Uncooked")


def _generated(network: Network, counts: Dict[str, int]) -> None:
    """Give the scheduler the items a static generate makes, ``counts[node]`` for each step."""
    # PDG keeps the items on the scheduler, which Submit finds by its name.
    commands: Dict[str, str] = {node.name: node.command for node in network.pdg_nodes}
    items: Dict[Any, List[Any]] = {
        _Item(name, i, commands[name]): [] for name, count in counts.items() for i in range(count)
    }
    scheduler: Any = network.scheduler()
    scheduler.dependency_graph = (items, {}, [])
    network.context.schedulers["deadlinecloudscheduler1"] = scheduler


def test_submit_generates_first_and_confirms_the_step_and_task_counts(scene: FakeHou) -> None:
    """The artist sees the job's real size before choosing Submit, and the generate blocks."""
    # The generate makes five tasks over three steps.
    network: Network = _network()
    _generated(network, {"sim": 2, "render": 2, "montage": 1})
    display: Any = network.topnet.displayNode()
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    assert display.generate_calls == [{"block": True}]
    assert "Steps: 3, tasks: 5" in scene.messages.calls[0][0]


def test_a_refusal_that_needs_items_is_shown_before_the_confirmation(scene: FakeHou) -> None:
    """Before, a step missing its items was only found in the cook, on the node's error flag."""
    # The generate leaves render with no items, and render is in the cook.
    network: Network = _network()
    _generated(network, {"sim": 2, "montage": 1})
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    ((text, buttons),) = scene.messages.calls
    assert "render:" in text and buttons == ()
    assert _cooks(network) == []


def test_the_generate_runs_after_the_save_and_leaves_the_saved_scene_saved(scene: FakeHou) -> None:
    """A generate that marked the scene changed would make Submit refuse the scene it just saved."""
    # An unsaved scene, saved by Submit, then generated.
    scene.hip_file.unsaved = True
    network: Network = _network()
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    display: Any = network.topnet.displayNode()
    display.hip_file = scene.hip_file
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    assert scene.hip_file.saves == 1 and display.generate_calls == [{"block": True}]
    assert scene.hip_file.hasUnsavedChanges() is False
    assert _parm_writes(network) == []


def test_a_network_over_a_shape_quota_is_refused_before_it_is_generated(scene: FakeHou) -> None:
    """Generating a graph the farm can't take costs time and memory for nothing."""
    # 201 steps in a row, one more than the service's 200 steps per job.
    spec: List[Tuple[Any, ...]] = [
        (f"step{i}", "hython run.py", "Processor", [f"step{i - 1}"] if i else [])
        for i in range(201)
    ]
    network: Network = build_network(spec, scheduler_parms={"working_dir": "$HIP"})
    display: Any = network.topnet.displayNode()
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert display.generate_calls == [] and _cooks(network) == []
    assert "201 steps per job" in scene.messages.calls[0][0]
    assert scene.messages.options[0]["severity"] == "Error"


def test_a_step_over_the_task_quota_stops_submits_generate_and_is_shown(scene: FakeHou) -> None:
    """The generate stops at the first step that's too big, so no confirmation follows it."""
    # sim generates one task more than a step may have, while Submit generates.
    network: Network = _network()
    display: Any = network.topnet.displayNode()

    def generate() -> None:
        network.node("sim").workItems = [None] * 10_001
        network.context.emit("NodeGenerated", "sim")

    display.on_generate = generate
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    ((text, buttons),) = scene.messages.calls
    assert "sim: 10,001 tasks per step" in text and buttons == ()
    assert scene.messages.options[0]["severity"] == "Error"
    assert network.context.canceled.wait(5) and _cooks(network) == []
    # The count stops with the generate, so a later cook isn't counted by it.
    assert network.context.handlers == []


def _with_requirements(network: Network, **parms: Any) -> None:
    """Set the Host Requirements tab to custom with ``parms``, the rest at the HDA's defaults."""
    # Every number at -1 and both lists empty, as a new node has them.
    values: Dict[str, Any] = {"hr_mode": "custom", "hr_os": "", "hr_cpu_arch": ""}
    for field in ("vcpu", "memory", "gpu", "gpu_memory", "scratch"):
        values[f"hr_{field}_min"] = -1
        values[f"hr_{field}_max"] = -1
    values.update({"hr_amounts": 0, "hr_attributes": 0}, **parms)
    for name, value in values.items():
        network.scheduler_node.set_parm(name, value)


def test_the_confirmation_lists_the_host_requirements(scene: FakeHou) -> None:
    """The artist sees which workers can take the job before choosing Submit."""
    # Linux workers with at least one GPU.
    network: Network = _network()
    _with_requirements(network, hr_os="linux", hr_gpu_min=1)
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    text: str = scene.messages.calls[0][0]
    assert "Host requirements: attr.worker.os.family any of linux, amount.worker.gpu min 1" in text


def test_the_confirmation_says_any_worker_when_there_are_no_requirements(scene: FakeHou) -> None:
    """Without the line an artist can't tell a job for any worker from one whose tab was ignored."""
    network: Network = _network()
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    assert "Host requirements: any worker" in scene.messages.calls[0][0]


def test_an_invalid_host_requirement_is_shown_and_nothing_cooks(scene: FakeHou) -> None:
    """The service would refuse the job only after the scene was uploaded."""
    # A GPU minimum above its maximum.
    network: Network = _network()
    _with_requirements(network, hr_gpu_min=4, hr_gpu_max=2)
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    ((text, buttons),) = scene.messages.calls
    assert "hr_gpu_min" in text and buttons == ()
    assert _cooks(network) == []


def test_the_confirmation_counts_the_extra_attachments(scene: FakeHou, tmp_path: Path) -> None:
    """The artist sees that the listed files and folders go with the job before choosing Submit."""
    # One file, one folder and two output folders on the tab.
    network: Network = _network()
    node: Any = network.scheduler_node
    lut: Path = tmp_path / "lut.cube"
    lut.write_text("")
    (tmp_path / "library").mkdir()
    for list_parm, entry, values in (
        ("input_filenames", "input_filenames", [str(lut)]),
        ("input_directories", "input_dir", [str(tmp_path / "library")]),
        ("output_directories", "output_dir", [str(tmp_path / "a"), str(tmp_path / "b")]),
    ):
        node.set_parm(list_parm, len(values))
        for index, value in enumerate(values, start=1):
            node.set_parm(f"{entry}{index}", value, kind="FileReference")
    assert submit(node, get_setting=_workstation) is True
    text: str = scene.messages.calls[0][0]
    assert "Extra attachments: 1 input file(s), 1 input folder(s), 2 output folder(s)" in text


def test_the_confirmation_says_when_there_are_no_extra_attachments(scene: FakeHou) -> None:
    """A line that appears only sometimes is easy to miss, so it is always shown."""
    network: Network = _network()
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    assert "Extra attachments: none" in scene.messages.calls[0][0]


def test_a_missing_listed_file_is_shown_before_the_confirmation(
    scene: FakeHou, tmp_path: Path
) -> None:
    """Before, the typo surfaced only once the cook had started, on the node's error flag."""
    # One listed file that isn't on disk.
    network: Network = _network()
    node: Any = network.scheduler_node
    node.set_parm("input_filenames", 1)
    node.set_parm("input_filenames1", str(tmp_path / "nowhere.cube"), kind="FileReference")
    assert submit(node, get_setting=_workstation) is False
    ((text, buttons),) = scene.messages.calls
    assert "input_filenames1" in text and buttons == ()
    assert _cooks(network) == []


def test_one_extra_input_file_alone_is_counted(scene: FakeHou, tmp_path: Path) -> None:
    """With no output folder listed, the line must still count the input the artist added."""
    # One listed file, and nothing else on the tab.
    network: Network = _network()
    node: Any = network.scheduler_node
    lut: Path = tmp_path / "lut.cube"
    lut.write_text("")
    node.set_parm("input_filenames", 1)
    node.set_parm("input_filenames1", str(lut), kind="FileReference")
    assert submit(node, get_setting=_workstation) is True
    text: str = scene.messages.calls[0][0]
    assert "Extra attachments: 1 input file(s), 0 input folder(s), 0 output folder(s)" in text


def test_after_generating_a_read_no_step_writes_and_no_disk_holds_is_refused(
    scene: FakeHou,
) -> None:
    """With the items generated, a missing file outside the job is known, so Submit names it."""
    # render reads a literal file that no step writes and the disk doesn't hold.
    network: Network = _network()
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    network.node("render").hom.set_parm("file", "/elsewhere/tree.bgeo.sc", kind="FileReference")
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert "/obj/topnet1/render parm 'file'" in scene.messages.calls[0][0]
    assert _cooks(network) == []


def test_an_unconnected_input_slot_does_not_hide_a_cook_with_no_step(scene: FakeHou) -> None:
    """Houdini lists an empty input slot as None, which has no path and names no node."""
    # The wedge, with one empty input slot, is the display node.
    network: Network = _network()
    wedge: Any = network.node("wedge1").hom
    wedge._inputs = [None]
    network.topnet.display = wedge
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert "No node in this cook runs a command on a worker" in scene.messages.calls[0][0]


def test_after_generating_a_step_read_only_by_its_files_is_still_consumed(scene: FakeHou) -> None:
    """A step joined to its reader only by a file read must not count as a leaf, or its lost outputs pass."""
    # sim and render share no wire, and render reads the file sim writes outside the job.
    network: Network = build_network(
        [
            ("sim", "hython sim.py", "Processor", []),
            ("render", "hython render.py", "Processor", []),
        ],
        scheduler_parms={"working_dir": "$HIP", "auto_save_hip": 0},
    )
    network.node("render").hom.set_parm("file", "/elsewhere/sim/out.bgeo.sc", kind="FileReference")
    _generated(network, {"sim": 1, "render": 1})
    graph: Any = network.context.schedulers["deadlinecloudscheduler1"].dependency_graph[0]
    (sim_item,) = [item for item in graph if item.node.name == "sim"]
    sim_item.expectedOutputFiles = [("/elsewhere/sim/out.bgeo.sc", "file/geo")]
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert "sim:" in scene.messages.calls[0][0] and "outside the working directory" in (
        scene.messages.calls[0][0]
    )


def test_a_node_whose_pdg_name_differs_from_its_name_is_named_as_pdg_names_it(
    scene: FakeHou,
) -> None:
    """PDG names a node inside a subnet differently, and the cook set check reads PDG's names."""
    network: Network = _network()
    network.node("wedge1").name = "subnet1_wedge1"
    network.topnet.display = network.node("subnet1_wedge1").hom
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert "No node in this cook runs a command on a worker" in scene.messages.calls[0][0]
