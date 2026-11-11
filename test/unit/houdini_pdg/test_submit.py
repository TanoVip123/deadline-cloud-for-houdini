# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The scheduler node's Submit button, against small fakes of ``hou`` and ``pdg``."""

from __future__ import annotations

import os
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from .houdini_fakes import (
    PYRO_BONFIRE,
    COOKING_EXAMPLE,
    FakeHou,
    FakeNetwork,
    FakeNode,
    build_network,
    install_hou,
)

from deadline.houdini_pdg.houdini.host import submit


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


def _network() -> FakeNetwork:
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


def _cooks(network: FakeNetwork) -> List[Dict[str, Any]]:
    """The cooks the display node was asked for."""
    display: Any = network.topnet.displayNode()
    return list(display.cook_calls)


def _parm_writes(network: FakeNetwork) -> List[Any]:
    """Every value written to any scheduler node parm."""
    return [value for parm in network.scheduler_node.parms() for value in parm.sets]


def test_a_saved_scene_submits_after_one_confirmation_and_stays_saved(scene: FakeHou) -> None:
    """A saved scene gets no save prompt, and Submit writes no parm, so the scene stays saved."""
    network: FakeNetwork = _network()
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    assert [buttons for _, buttons in scene.messages.calls] == [("Submit", "Cancel")]
    assert scene.hip_file.hasUnsavedChanges() is False and scene.hip_file.saves == 0
    assert _parm_writes(network) == []
    assert _cooks(network) == [{"block": False}]


@pytest.mark.parametrize("answers", [[2], [0, 1], [0, None]])
def test_cancel_cooks_nothing_and_writes_nothing(
    scene: FakeHou, answers: List[Optional[int]]
) -> None:
    """Cancel at the save prompt or at the confirmation leaves the scene and the node as they were.

    Closing the confirmation is a cancel too, so a stray click never submits a job.
    """
    scene.hip_file.unsaved = True
    scene.messages.answers = list(answers)
    network: FakeNetwork = _network()
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert _cooks(network) == []
    assert _parm_writes(network) == []


def test_enter_picks_save_on_the_save_prompt_and_submit_on_the_confirmation(
    scene: FakeHou,
) -> None:
    """Enter must never pick Always Save, which writes a parm, or Cancel instead of Submit."""
    # Submit an unsaved scene, then name the button Enter and the close box pick on each dialog.
    scene.hip_file.unsaved = True
    network: FakeNetwork = _network()
    submit(network.scheduler_node, get_setting=_workstation)
    picked: List[Tuple[str, str]] = [
        (buttons[options["default_choice"]], buttons[options["close_choice"]])
        for (_, buttons), options in zip(scene.messages.calls, scene.messages.options)
    ]
    assert picked == [("Save", "Cancel"), ("Submit", "Cancel")]


@pytest.mark.parametrize("default_scheduler", ["localscheduler", None])
def test_a_network_that_does_not_cook_on_this_scheduler_is_refused(
    scene: FakeHou, default_scheduler: Optional[str]
) -> None:
    """Cooking the display node uses the network's default scheduler, which could be another one.

    A scheduler inside a TOP subnet has a parent with no such parm, and Submit must say so, not crash.
    """
    # The network names another default scheduler, or has no such parm.
    network: FakeNetwork = _network()
    if default_scheduler is None:
        network.topnet.remove_parm("topscheduler")
    else:
        network.topnet.set_parm("topscheduler", default_scheduler)
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert _cooks(network) == []
    assert "Default TOP Scheduler" in scene.messages.calls[0][0]


def test_the_confirmation_shows_the_jobs_settings_and_any_warning(scene: FakeHou) -> None:
    """The artist confirms what will run, and a waived refusal is shown, not hidden."""
    network: FakeNetwork = build_network(
        COOKING_EXAMPLE,
        scheduler_parms={"working_dir": "$HIP", "validate_graph": 0, "priority": 70},
    )
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    text: str = scene.messages.calls[0][0]
    # With no name parm the job takes the scene's file name.
    assert "Job name: shot.hip" in text and "Priority: 70" in text
    assert "Farm: farm-1" in text and "Queue: queue-1" in text
    assert "python1" in scene.messages.options[0]["details"]


def test_submitting_a_display_node_with_no_step_above_it_is_refused(scene: FakeHou) -> None:
    """With the wedge displayed, the cook would run only in-process nodes and make no job."""
    # Display the wedge, then Submit, which must refuse before cooking.
    network: FakeNetwork = _network()
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


def _generated(network: FakeNetwork, counts: Dict[str, int]) -> None:
    """Give the scheduler the items a static generate makes, ``counts[node]`` for each step."""
    # PDG keeps the items on the scheduler, which Submit finds by its name.
    commands: Dict[str, str] = {node.name: node.command for node in network.pdg_nodes}
    items: Dict[Any, List[Any]] = {
        _Item(name, i, commands[name]): [] for name, count in counts.items() for i in range(count)
    }
    scheduler: Any = network.scheduler()
    scheduler.dependency_graph = (items, {}, [])
    network.context.schedulers["deadlinecloudscheduler1"] = scheduler


def test_a_network_over_a_shape_quota_is_refused_before_it_is_generated(scene: FakeHou) -> None:
    """Generating a graph the farm can't take costs time and memory for nothing."""
    # 201 steps in a row, one more than the service's 200 steps per job.
    spec: List[Tuple[Any, ...]] = [
        (f"step{i}", "hython run.py", "Processor", [f"step{i - 1}"] if i else [])
        for i in range(201)
    ]
    network: FakeNetwork = build_network(spec, scheduler_parms={"working_dir": "$HIP"})
    display: Any = network.topnet.displayNode()
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert ("generate", True) not in display.work_item_calls and _cooks(network) == []
    assert "201 steps per job" in scene.messages.calls[0][0]
    assert scene.messages.options[0]["severity"] == "Error"


def test_a_step_over_the_task_quota_stops_submits_generate_and_is_shown(scene: FakeHou) -> None:
    """The generate stops at the first step that's too big, so no confirmation follows it."""
    # sim generates one task more than a step may have, while Submit generates.
    network: FakeNetwork = _network()
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


def _with_requirements(network: FakeNetwork, **parms: Any) -> None:
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
    network: FakeNetwork = _network()
    _with_requirements(network, hr_os="linux", hr_gpu_min=1)
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    text: str = scene.messages.calls[0][0]
    assert "Host requirements: attr.worker.os.family any of linux, amount.worker.gpu min 1" in text


def test_the_confirmation_counts_the_extra_attachments(scene: FakeHou, tmp_path: Path) -> None:
    """The artist sees that the listed files and folders go with the job before choosing Submit."""
    # One file, one folder and two output folders on the tab.
    network: FakeNetwork = _network()
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


def test_one_extra_input_file_alone_is_counted(scene: FakeHou, tmp_path: Path) -> None:
    """With no output folder listed, the line must still count the input the artist added."""
    # One listed file, and nothing else on the tab.
    network: FakeNetwork = _network()
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
    network: FakeNetwork = _network()
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    network.node("render").hom.set_parm("file", "/elsewhere/tree.bgeo.sc", kind="FileReference")
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert "/obj/topnet1/render parm 'file'" in scene.messages.calls[0][0]
    assert _cooks(network) == []


class _Unprintable:
    """A value Houdini hands back that can't be turned into text, as a broken node might."""

    def __str__(self) -> str:
        raise RuntimeError("unreadable")


def test_a_scene_walk_that_fails_after_the_generate_still_gets_a_verdict(
    scene: FakeHou,
) -> None:
    """With no scene reads, Submit must judge the generated items alone, not stop on the missing reads.

    One step's node can't be read, so the scene walk reports itself unavailable.
    """
    # render's node gives a name that can't be turned into text, after a full generate.
    network: FakeNetwork = _network()
    network.node("render").hom.name = lambda: _Unprintable()  # type: ignore[method-assign]
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    assert "Steps: 3, tasks: 3" in scene.messages.calls[0][0]


def test_submit_reads_the_items_with_partitions_expanded(scene: FakeHou) -> None:
    """Submit must read the same expanded graph the cook reads, or the two could judge different jobs.

    In Houdini 21.0 the expanded graph gave every step its items with no false refusal.
    """
    # Generate one item per step, Submit, and read which graph the scheduler was asked for.
    network: FakeNetwork = _network()
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    scheduler: Any = network.context.schedulers["deadlinecloudscheduler1"]
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    assert scheduler.calls == [("dependencyGraph", True)]


def test_a_network_with_no_pdg_graph_is_told_so_and_nothing_is_generated(scene: FakeHou) -> None:
    """Submit went on with no graph and raised a bare AttributeError, which reached the artist."""
    # The node has no graph context, and cooking its TOP nodes doesn't make one either.
    network: FakeNetwork = _network()
    network.scheduler_node.context = None
    network.scheduler_node.cook_creates = None
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert [text for text, _ in scene.messages.calls] == [
        "The TOP network has no PDG graph yet, so Submit can't check it. "
        "Cook the network's TOP nodes once, then submit again."
    ]
    display: Any = network.topnet.displayNode()
    assert ("generate", True) not in display.work_item_calls and _cooks(network) == []


def test_submit_dirties_the_items_keeping_files_before_it_generates(scene: FakeHou) -> None:
    """PDG keeps old items after a SOP parm outside the network changes, so Submit judged stale outputs.

    The fake regenerates sim's declared output from the SOP's parm only after a dirty, as
    PDG does, and the dirty must keep files on disk, or the artist's results are deleted.
    """
    # sim's items were generated with an output inside the job, before the artist's change.
    network: FakeNetwork = _network()
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    graph: Any = network.context.schedulers["deadlinecloudscheduler1"].dependency_graph[0]
    (sim_item,) = [item for item in graph if item.node.name == "sim"]
    sim_item.expectedOutputFiles = [(os.environ["HIP"] + "/geo/sim.bgeo.sc", "file/geo")]
    sop_file: Any = FakeNode("/obj/pyro/pyro_cache", parms={"file": "/elsewhere/sim.bgeo.sc"}).parm(
        "file"
    )
    display: Any = network.topnet.displayNode()

    # A generate reads the SOP's parm again only for items dirtied since the last generate.
    def generate() -> None:
        if display.work_item_calls and display.work_item_calls[-1][0] == "dirty":
            sim_item.expectedOutputFiles = [(sop_file.eval(), "file/geo")]

    display.on_generate = generate
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert display.work_item_calls == [("dirty", False), ("generate", True)]
    text: str = scene.messages.calls[0][0]
    assert "sim:" in text and "outside the working directory" in text
    assert _cooks(network) == []


class _ErrorTab:
    """The scheduler node's error tab, as Houdini 21.0's UI showed it.

    A ``cookError`` shows once ``hou.ui.triggerUpdate`` runs, and Submit's dirty of the
    display node's items clears what shows. ``events`` keeps the order of those calls and the popups.
    """

    def __init__(self, network: FakeNetwork, scene: FakeHou) -> None:
        self.shown: List[str] = []
        self.events: List[str] = []
        self.scheduler: Any = network.context.schedulers["deadlinecloudscheduler1"]
        self._read: int = 0
        scene.messages.on_show = lambda: self.events.append("popup")
        scene.on_update = self._update
        # Record the dirty, which clears the tab, and each cookError, in the order they run.
        display: Any = network.topnet.displayNode()
        dirty: Any = display.dirtyAllWorkItems
        cook_error: Any = self.scheduler.cookError

        def dirty_and_clear(remove_outputs: bool) -> None:
            dirty(remove_outputs)
            self.events.append("dirty")
            self.shown.clear()

        def record_cook_error(message: str) -> None:
            cook_error(message)
            self.events.append("cookError")

        display.dirtyAllWorkItems = dirty_and_clear
        self.scheduler.cookError = record_cook_error

    def _update(self) -> None:
        """Show every cookError made since the last update, as Houdini's UI update does."""
        # Record the update, then show the cookErrors after the last one this model showed.
        self.events.append("update")
        calls: List[Tuple[Any, ...]] = self.scheduler.calls
        self.shown.extend(c[1] for c in calls[self._read :] if c[0] == "cookError")
        self._read = len(calls)


def _refused_on_the_tab(scene: FakeHou) -> Tuple[FakeNetwork, _ErrorTab, str]:
    """Submit a network whose render step is an In-Process step, which is refused, and return its popup text."""
    # PDG has the scheduler from an earlier generate, and render becomes an In-Process step.
    network: FakeNetwork = _network()
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    tab: _ErrorTab = _ErrorTab(network, scene)
    network.node("render").hom.set_parm("pdg_cooktype", 1)
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    return network, tab, scene.messages.calls[-1][0]


def test_a_refusal_stays_on_the_error_tab_until_the_fixed_scenes_next_submit(
    scene: FakeHou,
) -> None:
    """The popup closes, so the artist needs the text on the node while fixing the scene.

    The text must show at once, through Houdini's UI update after ``cookError``, and
    Submit's next dirty must clear it, so a fixed scene shows no stale error.
    """
    # The first Submit is refused, and its popup text shows on the tab.
    network: FakeNetwork
    tab: _ErrorTab
    text: str
    network, tab, text = _refused_on_the_tab(scene)
    assert "The cook can't start." in text and tab.shown == [text]
    # The display node's items are only Submit's own dirty and generate.
    display: Any = network.topnet.displayNode()
    assert display.work_item_calls == [("dirty", False), ("generate", True)]
    # Fix render and Submit again.
    network.node("render").hom.remove_parm("pdg_cooktype")
    assert submit(network.scheduler_node, get_setting=_workstation) is True
    assert tab.shown == []


def test_a_cancel_at_the_save_prompt_leaves_the_error_tab_as_it_was(scene: FakeHou) -> None:
    """Cancel checks nothing, so it must neither clear the last error nor add one."""
    # A refused Submit, then a change to the scene and a cancel at the save prompt.
    network: FakeNetwork
    tab: _ErrorTab
    text: str
    network, tab, text = _refused_on_the_tab(scene)
    scene.hip_file.unsaved = True
    scene.messages.answers = [2]
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert tab.shown == [text]
    assert [call for call in tab.scheduler.calls if call[0] == "cookError"] == [("cookError", text)]


def test_a_stop_before_the_dirty_leaves_the_scheduler_node_alone(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """A bad tab value stops Submit before it dirties or generates the display node's items.

    Only the refusals found on the generated items go on the error tab, so only the popup
    shows, with a log line, even though PDG has the scheduler.
    """
    # PDG has the scheduler from an earlier generate, and the GPU minimum is above its maximum.
    network: FakeNetwork = _network()
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    tab: _ErrorTab = _ErrorTab(network, scene)
    _with_requirements(network, hr_gpu_min=4, hr_gpu_max=2)
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert "hr_gpu_min" in scene.messages.calls[0][0]
    assert network.scheduler_node.generate_calls == [] and tab.scheduler.calls == []
    assert scene.updates == 0 and "error tab" in caplog.text


def test_a_service_node_is_refused_from_its_cook_type_parm_and_replaces_the_tabs_text(
    scene: FakeHou,
) -> None:
    """The display node's generate starts PDG's service processes for a Service node, and was slow to refuse.

    PDG keeps the last generate's cook type, so the parm must decide. Neither node may generate,
    and the dirty must come before the cookError, or the tab keeps the earlier Submit's text.
    """
    # An earlier Submit put an In-Process refusal on the tab.
    network: FakeNetwork
    tab: _ErrorTab
    network, tab, _first = _refused_on_the_tab(scene)
    tab.events.clear()
    # PDG still says Out-of-Process from that generate, and the artist sets render to Service.
    network.node("render").cookType = "OutOfProcess"
    network.node("render").hom.set_parm("pdg_cooktype", 3)
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    text: str = scene.messages.calls[-1][0]
    assert text == (
        "The cook can't start.\n\nThe graph was refused (1 refusal(s)):\n- render: This node's "
        "work items cook in a long-running service process, which this scheduler does not "
        "start. Set the node's 'pdg_cooktype' parm to Out Of Process, or cook this network "
        "with another scheduler."
    )
    # Only the new text shows, and this Submit only dirtied the display node's items.
    assert tab.shown == [text] and tab.events == ["dirty", "cookError", "popup", "update"]
    display: Any = network.topnet.displayNode()
    assert display.work_item_calls[2:] == [("dirty", False)]
    assert network.scheduler_node.generate_calls == []


def test_a_quota_stop_during_the_display_generate_leaves_the_scheduler_node_alone(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """A task quota cancels the display node's generate after Submit's dirty.

    A quota stop goes only in the popup, so Submit must leave the tab alone and log why it
    is empty.
    """
    # PDG has the scheduler, and sim generates one task more than a step may have.
    network: FakeNetwork = _network()
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    tab: _ErrorTab = _ErrorTab(network, scene)
    display: Any = network.topnet.displayNode()

    def generate() -> None:
        network.node("sim").workItems = [None] * 10_001
        network.context.emit("NodeGenerated", "sim")
        # PDG's generate ends when the cancel lands.
        network.context.canceled.wait(5)

    # Submit dirties, its generate stops at the quota, and the popup names the step.
    display.on_generate = generate
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert "sim: 10,001 tasks per step" in scene.messages.calls[0][0]
    assert display.work_item_calls[0] == ("dirty", False)
    assert network.scheduler_node.generate_calls == [] and tab.scheduler.calls == []
    assert scene.updates == 0 and "error tab" in caplog.text


@pytest.mark.parametrize(
    "cook_type, work_item_calls",
    [(1, [("dirty", False), ("generate", True)]), (3, [("dirty", False)])],
    ids=["in-process", "service"],
)
def test_with_no_scheduler_only_the_popup_shows(
    scene: FakeHou,
    caplog: pytest.LogCaptureFixture,
    cook_type: int,
    work_item_calls: List[Tuple[str, bool]],
) -> None:
    """PDG names no scheduler in a fresh session, and may name none even after a generate.

    Only the popup can show then, so the log must say why the tab is empty, with no UI update.
    """
    # PDG names no scheduler, and render is an In-Process step or a Service node.
    network: FakeNetwork = _network()
    network.node("render").hom.set_parm("pdg_cooktype", cook_type)
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert "The cook can't start." in scene.messages.calls[0][0]
    display: Any = network.topnet.displayNode()
    assert display.work_item_calls == work_item_calls
    assert network.scheduler_node.generate_calls == [] and scene.updates == 0
    assert "Could not put the error on" in caplog.text


def test_in_the_ui_each_refused_submit_shows_its_own_text_on_the_error_tab(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """In Houdini's UI a cookError shows on the tab only once the UI updates.

    A generate on the scheduler node made that update, but it can start PDG's processes, so
    Submit asks the UI to update after the popup. Without it the tab showed the previous text.
    """
    # PDG has the scheduler, and render is an In-Process step, which is refused after the generate.
    caplog.set_level("INFO")
    network: FakeNetwork = _network()
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    tab: _ErrorTab = _ErrorTab(network, scene)
    network.node("render").hom.set_parm("pdg_cooktype", 1)
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    first: str = scene.messages.calls[-1][0]
    assert tab.shown == [first] and tab.events[-2:] == ["popup", "update"]
    assert network.scheduler_node.generate_calls == []
    assert "Sent Submit's error to /obj/topnet1/deadlinecloudscheduler1's error tab." in caplog.text
    # A second refusal with different text replaces the first on the tab.
    network.node("montage").hom.set_parm("pdg_cooktype", 1)
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    second: str = scene.messages.calls[-1][0]
    assert second != first and tab.shown == [second]
    assert network.scheduler_node.generate_calls == []


@pytest.mark.parametrize("cook_type", [1, 3], ids=["in-process", "service"])
def test_in_hython_the_error_is_sent_to_the_tab_with_no_ui_update(
    scene: FakeHou, caplog: pytest.LogCaptureFixture, cook_type: int
) -> None:
    """hython has no ``hou.ui`` to update, so the update is skipped and the log says why."""
    # PDG has the scheduler, and render is an In-Process step or a Service node.
    caplog.set_level("INFO")
    scene.ui = False
    network: FakeNetwork = _network()
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    tab: _ErrorTab = _ErrorTab(network, scene)
    network.node("render").hom.set_parm("pdg_cooktype", cook_type)
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert "cookError" in tab.events and scene.updates == 0
    assert network.scheduler_node.generate_calls == []
    assert "error tab, which may update late with no UI." in caplog.text


def test_in_the_ui_a_stop_before_the_display_generate_finishes_skips_the_error_tab(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the refusals found on the generated items go on the error tab.

    Moving cookError before the popup must not move it ahead of that guard, so a stop before
    the dirty, or a quota stop during the generate, makes no cookError and no UI update.
    """
    caplog.set_level("INFO")
    network: FakeNetwork = _network()
    _generated(network, {"sim": 1, "render": 1, "montage": 1})
    tab: _ErrorTab = _ErrorTab(network, scene)
    display: Any = network.topnet.displayNode()
    # First, a bad tab value stops Submit before the dirty.
    _with_requirements(network, hr_gpu_min=4, hr_gpu_max=2)
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert "hr_gpu_min" in scene.messages.calls[-1][0] and display.work_item_calls == []

    # Second, with the tab fixed, sim passes the task quota during the display generate.
    def generate() -> None:
        network.node("sim").workItems = [None] * 10_001
        network.context.emit("NodeGenerated", "sim")
        network.context.canceled.wait(5)

    _with_requirements(network, hr_gpu_min=-1, hr_gpu_max=-1)
    display.on_generate = generate
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert "sim: 10,001 tasks per step" in scene.messages.calls[-1][0]
    # Neither stop touched the scheduler node, and each logged the skip and not the set.
    assert network.scheduler_node.generate_calls == [] and tab.scheduler.calls == []
    assert scene.updates == 0 and tab.shown == []
    assert caplog.text.count("Could not put the error on") == 2
    assert "Sent Submit's error to" not in caplog.text
