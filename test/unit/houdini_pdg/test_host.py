# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The Houdini host and the pre-cook checks, against small fakes of ``hou`` and ``pdg``."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from .houdini_fakes import (
    COOKING_EXAMPLE,
    PYRO_BONFIRE,
    ROUTED_EXAMPLE,
    SCHEDULER_PATH,
    FakeHou,
    FakeNode,
    FakePdgNode,
    FakePyScheduler,
    Network,
    build_network,
    install_hou,
    main_thread_runner,
)

from deadline.client.exceptions import UserInitiatedCancel
from deadline.houdini_pdg.errors import CookSetupError, PayloadTransferError
from deadline.houdini_pdg.graph import NodeInfo
from deadline.houdini_pdg.houdini import host as host_module
from deadline.houdini_pdg.houdini.host import (
    HoudiniPdgHost,
    PreCookChecks,
    cook_set_names,
    find_scheduler_node,
    read_scheduler_settings,
)
from deadline.houdini_pdg.houdini.settings import SchedulerSettings
from deadline.houdini_pdg.ports import (
    ItemOutcome,
    ItemState,
    OutputFile,
    SceneFiles,
    WorkItemHandle,
)
from deadline.houdini_pdg.scene_reads import SceneReads
from deadline.houdini_pdg.validation import RefusalScenario, Verdict


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeHou:
    """A saved scene at ``<tmp>/scene/shot.hip``, with a UI, and no debug override."""
    hip: str = os.path.join(str(tmp_path), "scene", "shot.hip")
    os.makedirs(os.path.dirname(hip))
    monkeypatch.delenv("DEADLINE_PDG_DEBUG_LEVEL", raising=False)
    monkeypatch.setenv("HIP", os.path.dirname(hip))
    return install_hou(monkeypatch, hip)


def handle(item_id: int = 7, native: Any = None) -> WorkItemHandle:
    """A work item handle for item 7 of ``render``."""
    return WorkItemHandle(
        item_id=item_id, index=3, name="render_3", node_name="render", native=native
    )


class NativeItem:
    """A ``pdg.WorkItem`` that counts ``clearOutputFiles`` calls, or raises from it."""

    def __init__(self, raises: bool = False) -> None:
        self.cleared: int = 0
        self.raises: bool = raises

    def clearOutputFiles(self) -> None:
        if self.raises:
            raise RuntimeError("no such method on this PDG build")
        self.cleared += 1

    def saveJSONString(self, global_attributes: bool) -> str:
        return json.dumps({"name": "render_3", "global": global_attributes})


# -- notify, serialize, working directory -------------------------------------


def test_a_started_item_has_its_earlier_outputs_cleared_before_it_starts(scene: FakeHou) -> None:
    """An earlier cook's output records are cleared, so this cook's are the only ones.

    PDG keeps the old record through the start flag and adds the new one beside
    it, so a consumer would see each file twice.
    """
    order: List[str] = []
    native: NativeItem = NativeItem()
    scheduler: FakePyScheduler = build_network(PYRO_BONFIRE).scheduler()
    # Record the clear and the start in one list, so their order is checked.
    native.clearOutputFiles = lambda: order.append("clear")  # type: ignore[method-assign]
    scheduler.onWorkItemStartCook = lambda *args: order.append(  # type: ignore[method-assign]
        f"start{args}"
    )
    HoudiniPdgHost(scheduler).notify(handle(native=native), ItemOutcome(ItemState.STARTED))
    assert order == ["clear", "start(7, -1, True)"]


@pytest.mark.parametrize("native", [None, NativeItem(raises=True)])
def test_an_item_that_cannot_be_cleared_still_starts_cooking(scene: FakeHou, native: Any) -> None:
    """A record left behind repeats an old duplicate, while raising would lose the task."""
    scheduler: FakePyScheduler = build_network(PYRO_BONFIRE).scheduler()
    HoudiniPdgHost(scheduler).notify(handle(native=native), ItemOutcome(ItemState.STARTED))
    assert scheduler.calls == [("onWorkItemStartCook", 7, -1, True)]


def test_only_a_started_item_is_cleared(scene: FakeHou) -> None:
    """Success and failure report an outcome, and neither drops what the cook produced."""
    host: HoudiniPdgHost = HoudiniPdgHost(build_network(PYRO_BONFIRE).scheduler())
    for outcome in (
        ItemOutcome(ItemState.SUCCEEDED, (OutputFile("/proj/b.exr", ""),)),
        ItemOutcome(ItemState.FAILED),
    ):
        native: NativeItem = NativeItem()
        host.notify(handle(native=native), outcome)
        assert native.cleared == 0


def test_a_succeeded_item_reports_each_output_unchanged_then_success(scene: FakeHou) -> None:
    """PDG routes outputs by path and tag, so both must reach it as the worker reported them."""
    scheduler: FakePyScheduler = build_network(PYRO_BONFIRE).scheduler()
    outputs: Tuple[OutputFile, ...] = (
        OutputFile("/proj/geo/a.bgeo.sc", "file/geo"),
        OutputFile("/proj/b.exr", ""),
    )
    HoudiniPdgHost(scheduler).notify(
        handle(), ItemOutcome(ItemState.SUCCEEDED, outputs, duration_seconds=12.5)
    )
    assert scheduler.calls == [
        ("onWorkItemAddOutput", 7, -1, "/proj/geo/a.bgeo.sc", "file/geo", 0),
        ("onWorkItemAddOutput", 7, -1, "/proj/b.exr", "", 0),
        ("onWorkItemSucceeded", 7, -1, 12.5),
    ]


def test_a_failed_item_is_reported_failed(scene: FakeHou) -> None:
    """A failed task fails its work item, so PDG stops what depends on it."""
    scheduler: FakePyScheduler = build_network(PYRO_BONFIRE).scheduler()
    HoudiniPdgHost(scheduler).notify(handle(), ItemOutcome(ItemState.FAILED))
    assert scheduler.calls == [("onWorkItemFailed", 7, -1)]


_CHECKSUM_SCRIPT: str = """
import json, sys, types
sys.modules["hou"] = types.ModuleType("hou")
sys.modules["pdg"] = types.ModuleType("pdg")
from deadline.houdini_pdg.houdini.host import HoudiniPdgHost
from deadline.houdini_pdg.ports import ItemOutcome, ItemState, OutputFile, WorkItemHandle

class Scheduler:
    calls = []
    def onWorkItemAddOutput(self, *args):
        self.calls.append(list(args))
    def onWorkItemSucceeded(self, *args):
        pass

scheduler = Scheduler()
outputs = tuple(OutputFile(path, tag) for path, tag in json.loads(sys.argv[1]))
HoudiniPdgHost(scheduler).notify(
    WorkItemHandle(item_id=1, index=0, name="a", node_name="n"),
    ItemOutcome(ItemState.SUCCEEDED, outputs),
)
print(json.dumps(scheduler.calls))
"""


def test_the_output_checksum_is_the_same_in_every_process() -> None:
    """A checksum that changed between sessions would make PDG treat every output as changed."""
    # Run the report in two processes with different hash seeds.
    outputs: List[List[str]] = [["", ""], ["/a/b.exr", "file/image"], ["__PDG_DIR__/x.$F4", "f"]]
    outputs += [[f"/p/{chr(0x4E00 + i)}/{i}.vdb", f"t{i}"] for i in range(20)]
    src: str = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
    runs: List[Any] = []
    for seed in ("1", "2"):
        env: Dict[str, str] = dict(os.environ, PYTHONHASHSEED=seed)
        env["PYTHONPATH"] = os.pathsep.join([os.path.abspath(src), env.get("PYTHONPATH", "")])
        result: subprocess.CompletedProcess = subprocess.run(
            [sys.executable, "-c", _CHECKSUM_SCRIPT, json.dumps(outputs)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        runs.append(json.loads(result.stdout))
    assert runs[0] == runs[1]
    assert [call[2:4] for call in runs[0]] == outputs
    assert {call[4] for call in runs[0]} == {0}


def _host_with_item_log(log: Any) -> HoudiniPdgHost:
    """A host whose PDG graph holds work item 7, with ``log`` as its log."""
    network: Network = build_network(PYRO_BONFIRE)
    network.context.items[7] = types.SimpleNamespace(logMessages=log)
    return HoudiniPdgHost(network.scheduler())


def test_a_failure_reason_is_the_error_line_of_the_items_pdg_log(scene: FakeHou) -> None:
    """The ERROR line, without its time stamp, is the reason, and the last line is kept apart.

    The text is what Houdini 21.0 left on an FFmpeg Encode Video item whose codec
    its hffmpeg lacks.
    """
    log: str = '[23:28:39.073] ERROR: Video codec "libx264" is not available on this platform.\n\n'
    assert _host_with_item_log(log).failure_reason(7) == (
        'Video codec "libx264" is not available on this platform.',
        'ERROR: Video codec "libx264" is not available on this platform.',
    )


def test_a_log_without_an_error_line_gives_no_error_and_its_last_line(scene: FakeHou) -> None:
    """Without an ERROR line there is no reason, only the last line, so it is not called one."""
    log: str = "[01:00:00.000] Preparing\n[01:00:01.000] Generated\n"
    assert _host_with_item_log(log).failure_reason(7) == ("", "Generated")


def test_a_failure_reason_is_empty_when_pdg_has_no_log_or_no_item(scene: FakeHou) -> None:
    """A missing item or log must not raise, because the cook is reporting a failure already."""

    class Unreadable:
        def __str__(self) -> str:
            raise RuntimeError("deleted")

    assert _host_with_item_log(None).failure_reason(7) == ("", "")
    assert _host_with_item_log("x").failure_reason(8) == ("", "")
    assert _host_with_item_log(Unreadable()).failure_reason(7) == ("", "")
    broken: Any = types.SimpleNamespace(context=None)
    assert HoudiniPdgHost(broken).failure_reason(7) == ("", "")


def test_serialize_returns_the_json_pdg_writes_for_a_job(scene: FakeHou) -> None:
    """The payload is PDG's own JSON, with the global attributes a worker needs."""
    host: HoudiniPdgHost = HoudiniPdgHost(build_network(PYRO_BONFIRE).scheduler())
    assert json.loads(host.serialize(handle(native=NativeItem()))) == {
        "name": "render_3",
        "global": True,
    }


def test_an_item_without_its_pdg_object_cannot_be_serialized(scene: FakeHou) -> None:
    """With no PDG object there is no payload, and the error names the item."""
    host: HoudiniPdgHost = HoudiniPdgHost(build_network(PYRO_BONFIRE).scheduler())
    with pytest.raises(PayloadTransferError, match="render_3"):
        host.serialize(handle())


def test_the_working_directory_is_both_local_and_remote(scene: FakeHou) -> None:
    """PDG writes job scripts under the local one and names files by the remote one."""
    scheduler: FakePyScheduler = build_network(PYRO_BONFIRE).scheduler()
    HoudiniPdgHost(scheduler).set_working_dir("/proj/shot")
    assert scheduler.calls == [("setWorkingDir", "/proj/shot", "/proj/shot")]


# -- generation ---------------------------------------------------------------


def _watching(cook_set: Any = None) -> Tuple[HoudiniPdgHost, Network]:
    """A host watching the generation of a Pyro Bonfire network."""
    network: Network = build_network(PYRO_BONFIRE)
    host: HoudiniPdgHost = HoudiniPdgHost(network.scheduler())
    host.watch_generation(cook_set)
    return host, network


def test_generation_is_pending_until_each_step_announces_it(scene: FakeHou) -> None:
    """The cook waits for every step, so a snapshot taken early does not miss the wedges' items."""
    host, network = _watching()
    assert host.pending_generation() == ("montage", "render", "sim")
    # The wedge is not a step, so its announcement changes nothing.
    network.context.emit("NodeGenerated", "sim")
    network.context.emit("NodeGenerated", "wedge1")
    assert host.pending_generation() == ("montage", "render")
    network.context.emit("NodeGenerated", "render")
    network.context.emit("NodeGenerated", "montage")
    assert host.pending_generation() == ()


def test_only_the_steps_pdg_asked_for_are_awaited(scene: FakeHou) -> None:
    """A step outside the cook set never generates, so waiting for it would stall the cook."""
    host, network = _watching(frozenset({"wedge1", "sim"}))
    assert host.pending_generation() == ("sim",)
    network.context.emit("NodeGenerated", "sim")
    assert host.pending_generation() == ()


def test_a_step_with_a_queued_item_has_generated_without_announcing_it(scene: FakeHou) -> None:
    """A step generated before this cook may not announce itself again, but a queued item shows it is done."""
    host, network = _watching()
    queued: Any = types.SimpleNamespace(state=types.SimpleNamespace(name="Waiting"))
    stale: Any = types.SimpleNamespace(state=types.SimpleNamespace(name="Uncooked"))
    network.node("sim").workItems = [stale, queued]
    network.node("render").workItems = [stale]
    assert host.pending_generation() == ("montage", "render")


def test_stopping_removes_the_handler_and_is_safe_twice(scene: FakeHou) -> None:
    """A stop after a failed start, or a second stop, must not raise or leave a handler."""
    host, network = _watching()
    assert len(network.context.handlers) == 1
    host.stop_watching_generation()
    host.stop_watching_generation()
    assert network.context.handlers == []


# -- read_graph and read_scene ------------------------------------------------


def test_read_graph_reads_the_dependency_graph_once_and_keeps_the_cook_set(scene: FakeHou) -> None:
    """One read gives a consistent snapshot, and a node outside the cook set is marked so."""
    # The cook set names two of the five nodes.
    network: Network = build_network(PYRO_BONFIRE)
    network.context.cookSet = [network.node("sim"), network.node("render")]
    scheduler: FakePyScheduler = network.scheduler()
    infos: Sequence[NodeInfo] = HoudiniPdgHost(scheduler).read_graph()
    assert scheduler.calls == [("dependencyGraph", True)]
    by_name: Dict[str, NodeInfo] = {info.name: info for info in infos}
    assert by_name["render"].upstream == ("sim",)
    assert by_name["sim"].is_schedulable and not by_name["wedge1"].is_schedulable
    assert by_name["sim"].in_cook_set is True and by_name["montage"].in_cook_set is False


def test_read_graph_leaves_out_the_items_this_cook_will_not_schedule(scene: FakeHou) -> None:
    """An item PDG left Uncooked is not in this cook, so the job must not get a task for it."""

    class Item:
        """A ``pdg.WorkItem`` of ``sim`` in the given state. PDG's items are hashable keys."""

        def __init__(self, index: int, state: str) -> None:
            self.name: str = f"sim_{index}"
            self.index: int = index
            self.id: int = 100 + index
            self.node: Any = types.SimpleNamespace(name="sim")
            self.command: str = "hython sim.py"
            self.state: Any = types.SimpleNamespace(name=state)

    def item(index: int, state: str) -> Any:
        return Item(index, state)

    network: Network = build_network(PYRO_BONFIRE)
    scheduler: FakePyScheduler = network.scheduler()
    scheduler.dependency_graph = ({item(0, "Waiting"): [], item(1, "Uncooked"): []}, {}, [])
    infos: Sequence[NodeInfo] = HoudiniPdgHost(scheduler).read_graph()
    sim: NodeInfo = next(info for info in infos if info.name == "sim")
    assert [i.name for i in sim.items] == ["sim_0"]


def test_a_pdg_node_whose_job_script_cannot_be_read_is_not_a_step(scene: FakeHou) -> None:
    """A node being deleted raises on every read, and must not be waited for as a step."""

    class Deleted(FakePdgNode):
        @property
        def scriptInfo(self) -> Any:
            raise RuntimeError("deleted")

    network: Network = build_network(PYRO_BONFIRE)
    network.pdg_nodes.append(Deleted(name="gone", command="hython x.py"))
    host: HoudiniPdgHost = HoudiniPdgHost(network.scheduler())
    host.watch_generation(None)
    assert "gone" not in host.pending_generation()


def test_an_override_naming_this_scheduler_is_cleared_and_another_is_kept(scene: FakeHou) -> None:
    """An override that names this scheduler changes nothing, so only another scheduler is refused."""
    network: Network = build_network(PYRO_BONFIRE)
    network.node("sim").hom.set_parm("topscheduler", "../deadlinecloudscheduler1")
    network.node("render").hom.set_parm("topscheduler", "localscheduler")
    network.node("render").scheduler_name = "localscheduler"
    infos: Sequence[NodeInfo] = HoudiniPdgHost(network.scheduler()).read_graph()
    by_name: Dict[str, NodeInfo] = {info.name: info for info in infos}
    assert by_name["sim"].scheduler_override == ""
    assert by_name["render"].scheduler_override == "localscheduler"


def test_an_empty_or_unreadable_cook_set_is_unknown() -> None:
    """An empty set would leave every node out, so it reads as unknown, like an unreadable one."""
    assert cook_set_names([]) is None
    assert cook_set_names(None) is None
    assert cook_set_names(42) is None
    assert cook_set_names([types.SimpleNamespace(name="a"), "b"]) == {"a", "b"}


def test_a_reads_static_prefix_has_its_variables_expanded(
    scene: FakeHou, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A read that depends on a work item is matched by its folder, so ``$HIP`` there must be expanded.

    An unknown variable expands to nothing in Houdini, which would shorten the path, so it gives no prefix.
    """
    network: Network = build_network(PYRO_BONFIRE)
    hom: Any = network.node("render").hom
    hom.set_parm("file", "$HIP/geo/`@pdg_index`.bgeo", kind="FileReference")
    hom.set_parm("other", "${NO_SUCH_VARIABLE_X}/geo/`@pdg_index`.bgeo", kind="FileReference")
    reads: SceneReads = HoudiniPdgHost(network.scheduler()).read_scene()
    prefixes: Dict[str, str] = {read.parm: read.static_prefix for read in reads.reads}
    assert prefixes["file"] == os.path.join(str(tmp_path / "scene"), "geo") + "/"
    assert prefixes["other"] == ""


def test_read_scene_walks_every_top_node_including_those_that_are_not_steps(
    scene: FakeHou,
) -> None:
    """A node that is not a step can read a step's file, so its file parms are read too."""
    network: Network = build_network(PYRO_BONFIRE)
    network.node("wedge1").hom.set_parm("file", "$HIP/wedge.json", kind="FileReference")
    network.node("sim").hom.set_parm("file", "$HIP/sim.bgeo", kind="FileReference")
    reads: SceneReads = HoudiniPdgHost(network.scheduler()).read_scene()
    assert {(read.step, read.parm) for read in reads.reads} == {("wedge1", "file"), ("sim", "file")}


# -- scene files --------------------------------------------------------------


def _touch(*parts: str) -> str:
    """Create an empty file, and its folders, and return its path."""
    path: str = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return path


def _reference(node: FakeNode, parm: str, value: str, scene: FakeHou) -> None:
    """Give ``node`` a file parm holding ``value``, and list it in ``hou.fileReferences``."""
    scene.file_references.append((node.set_parm(parm, value, kind="FileReference"), value))


def test_scene_files_are_the_saved_hip_and_the_expanded_assets_that_exist(
    scene: FakeHou, tmp_path: Path
) -> None:
    """Workers need real paths and every frame of a sequence, and nothing that doesn't exist."""
    folder: str = str(tmp_path / "scene")
    for name in ("tex.exr", "frame.0001.exr", "frame.0002.exr"):
        _touch(folder, name)
    os.makedirs(os.path.join(folder, "caches"))
    _touch(folder, "lib.hda")
    network: Network = build_network(PYRO_BONFIRE)
    sop: FakeNode = network.topnet.add(FakeNode("/obj/geo1/file1", category="Sop"))
    _reference(sop, "tex", "$HIP/tex.exr", scene)
    _reference(sop, "seq", "$HIP/frame.$F4.exr", scene)
    _reference(sop, "missing", "$HIP/missing.exr", scene)
    _reference(sop, "cache", "$HIP/caches", scene)
    # An installed asset library is held by no parm.
    scene.file_references += [(None, "$HIP/lib.hda"), (None, "opdef:/Sop/x")]
    host: HoudiniPdgHost = HoudiniPdgHost(network.scheduler())

    files: SceneFiles = host.scene_files()

    assert files.hip_file == os.path.join(folder, "shot.hip")
    expected: List[str] = ["tex.exr", "lib.hda", "frame.0001.exr", "frame.0002.exr"]
    assert files.assets == tuple(sorted(os.path.join(folder, name) for name in expected))
    assert files.directories == (os.path.join(folder, "caches"),)
    # One scan per cook: the cook and the submitter share it.
    assert host.scene_files() is files


def test_work_item_expressions_and_relative_paths_are_not_globbed(
    scene: FakeHou, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """With a work item active, the ROP's scan keeps `` `@pdg_input` ``; it must not become ``cwd/*``.

    A relative path would resolve against Houdini's current directory, which has
    nothing to do with the scene.
    """
    folder: str = str(tmp_path / "scene")
    _touch(folder, "frame.0001.exr")
    # A step's earlier output, which the wedge expression would glob into.
    _touch(folder, "render", "v0", "f.0001.jpg")
    # Houdini's current directory, standing in for the artist's home directory.
    cwd: Path = tmp_path / "home"
    (cwd / "Library").mkdir(parents=True)
    (cwd / "frame.0001.exr").write_text("")
    monkeypatch.chdir(cwd)
    network: Network = build_network(PYRO_BONFIRE)
    sop: FakeNode = network.topnet.add(FakeNode("/obj/geo1/file1", category="Sop"))
    # The scan keeps a reference only when it evaluates to a file, so each one does.
    for index, value in enumerate(
        ("`@pdg_input`", "$HIP/render/v`@wedgeindex`/f.$F4.jpg", "frame.$F4.exr")
    ):
        parm: Any = sop.set_parm(f"f{index}", value, kind="FileReference")
        parm.evalAsString = lambda: os.path.join(folder, "frame.0001.exr")
        scene.file_references.append((parm, value))
    _reference(sop, "real", "$HIP/frame.$F4.exr", scene)
    host: HoudiniPdgHost = HoudiniPdgHost(network.scheduler())

    with caplog.at_level(logging.INFO):
        files: SceneFiles = host.scene_files()

    assert files.assets == (os.path.join(folder, "frame.0001.exr"),)
    assert files.directories == ()
    assert "Not attaching '`@pdg_input`'" in caplog.text


def test_an_item_file_token_with_a_fake_home_is_not_attached(
    scene: FakeHou, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `` `@pdg_input` `` folder parm evaluates to a real folder under a work item, but is left out.

    The step that writes the file delivers it, and the home guard must not fire on
    a reference that is never attached.
    """
    # A home folder, and a token parm that evaluates to it while a work item is active.
    home: Path = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    network: Network = build_network(PYRO_BONFIRE)
    sop: FakeNode = network.topnet.add(FakeNode("/obj/geo1/file1", category="Sop"))
    parm: Any = sop.set_parm("dir", "`@pdg_input`", kind="FileReference")
    parm.evalAsString = lambda: str(home)
    scene.file_references.append((parm, "`@pdg_input`"))
    files: SceneFiles = HoudiniPdgHost(network.scheduler()).scene_files()
    assert files.directories == () and files.assets == ()


def test_a_failed_asset_scan_still_returns_the_hip(scene: FakeHou, caplog: Any) -> None:
    """The cook can still run without the assets, so it goes on, and the warning says what may be missing."""
    scene.scan_error = RuntimeError("no scene")
    host: HoudiniPdgHost = HoudiniPdgHost(build_network(PYRO_BONFIRE).scheduler())
    with caplog.at_level(logging.WARNING):
        files: SceneFiles = host.scene_files()
    assert files.hip_file.endswith("shot.hip") and files.assets == ()
    assert "may be missing on the workers" in caplog.text


def test_the_integrations_own_libraries_are_not_attached_and_expanded_libraries_are_directories(
    scene: FakeHou, tmp_path: Path
) -> None:
    """A worker never cooks the scheduler or the ROP, and job attachments refuses a folder listed as a file."""
    # Both integration libraries are stored expanded, as folders, next to two scene libraries.
    folder: str = str(tmp_path / "scene")
    plugin: str = str(tmp_path / "plugin")
    for library in ("deadline_cloud_scheduler.hda", "deadline_cloud.hda"):
        os.makedirs(os.path.join(plugin, library))
    os.makedirs(os.path.join(folder, "expanded.hda"))
    _touch(folder, "packed.hda")
    scene.file_references = [
        (None, os.path.join(plugin, "deadline_cloud_scheduler.hda")),
        (None, os.path.join(plugin, "deadline_cloud.hda")),
        (None, "$HIP/expanded.hda"),
        (None, "$HIP/packed.hda"),
    ]
    scene.libraries = {
        "deadlinecloudscheduler": os.path.join(plugin, "deadline_cloud_scheduler.hda"),
        "deadline_cloud": os.path.join(plugin, "deadline_cloud.hda"),
    }
    files: SceneFiles = HoudiniPdgHost(build_network(PYRO_BONFIRE).scheduler()).scene_files()
    assert files.assets == (os.path.join(folder, "packed.hda"),)
    assert files.directories == (os.path.join(folder, "expanded.hda"),)


# -- the home guard -----------------------------------------------------------


def _home_scene(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parm_value: str
) -> Tuple[HoudiniPdgHost, str]:
    """A scene saved directly in a fake home directory, with one folder parm holding ``parm_value``."""
    # Both home variables name the folder, as on Windows and in a shell on any system.
    home: str = str(tmp_path / "home")
    os.makedirs(home)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    hip: str = os.path.join(home, "shot.hip")
    open(hip, "w").close()
    monkeypatch.setenv("HIP", home)
    scene: FakeHou = install_hou(monkeypatch, hip)
    network: Network = build_network(PYRO_BONFIRE)
    sop: FakeNode = network.topnet.add(FakeNode("/obj/geo1/file1", category="Sop"))
    _reference(sop, "cachedir", parm_value, scene)
    return HoudiniPdgHost(network.scheduler()), home


@pytest.mark.parametrize("parm_value", ["$HIP", "$HOME"])
def test_a_folder_parm_that_expands_to_home_is_refused_naming_it_and_the_parm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parm_value: str
) -> None:
    """Attaching the home directory would hash all of it before submitting, so the cook would hang.

    ``$HIP`` reaches home when the ``.hip`` is saved directly in it.
    """
    host, home = _home_scene(monkeypatch, tmp_path, parm_value)
    with pytest.raises(CookSetupError) as raised:
        host.scene_files()
    assert home in str(raised.value)
    assert "/obj/geo1/file1/cachedir" in str(raised.value)


def test_a_folder_parm_that_holds_home_or_is_the_root_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A folder above home, or a filesystem root, holds the home directory too."""
    for value in (str(tmp_path), os.path.abspath(os.sep)):
        host, _ = _home_scene(
            monkeypatch, tmp_path / value.strip(os.sep).replace(os.sep, "_"), value
        )
        with pytest.raises(CookSetupError, match="cachedir"):
            host.scene_files()


def test_a_folder_below_home_is_attached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only home and above are refused. A project folder inside home is what most scenes use."""
    host, home = _home_scene(monkeypatch, tmp_path, "$HIP/caches")
    os.makedirs(os.path.join(home, "caches"))
    assert host.scene_files().directories == (os.path.join(home, "caches"),)


def test_the_home_comparison_ignores_case_where_the_filesystem_does() -> None:
    """Windows paths ignore case, so ``C:\\Users\\Ana`` and ``c:\\users\\ana`` are one folder."""
    home: str = os.path.normcase(os.path.abspath(os.path.join(os.sep, "Users", "Ana")))
    spelled: str = os.path.join(os.sep, "USERS", "ANA") if os.name == "nt" else home
    assert host_module._is_unbounded(spelled, [home])
    assert not host_module._is_unbounded(os.path.join(home, "proj"), [home])


# -- settings -----------------------------------------------------------------


def _settings_node(**parms: Any) -> FakeNode:
    """A scheduler node with the given parms."""
    return FakeNode(SCHEDULER_PATH, parms=parms)


def test_the_working_directory_is_expanded_and_absolute(scene: FakeHou, tmp_path: Path) -> None:
    """Workers map outputs by this path, so it must be a real, normalized one."""
    settings: SchedulerSettings = read_scheduler_settings(
        _settings_node(working_dir="$HIP/../out/./", debug_level=1)
    )
    assert settings.working_dir == str(tmp_path / "out")
    assert settings.debug_level == 1


def test_a_relative_working_directory_becomes_absolute(scene: FakeHou) -> None:
    """A relative path would mean a different folder on every worker."""
    assert os.path.isabs(read_scheduler_settings(_settings_node(working_dir="rel/dir")).working_dir)


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_working_directory_names_the_parm(scene: FakeHou, value: str) -> None:
    """The artist needs to know which parm to fix."""
    with pytest.raises(CookSetupError, match="'working_dir'"):
        read_scheduler_settings(_settings_node(working_dir=value))


def test_a_working_directory_with_an_unset_variable_names_the_parm(scene: FakeHou) -> None:
    """Houdini leaves an unknown variable in place, which no worker can resolve."""
    with pytest.raises(CookSetupError, match="'working_dir'.*NO_SUCH_VARIABLE_X"):
        read_scheduler_settings(_settings_node(working_dir="$NO_SUCH_VARIABLE_X/out"))


def test_an_unreadable_parm_names_the_parm(scene: FakeHou) -> None:
    """A broken expression must be reported as a setting problem, not as a crash."""
    with pytest.raises(CookSetupError, match="'working_dir'.*bad expression"):
        read_scheduler_settings(_settings_node(working_dir=RuntimeError("bad expression")))


def test_absent_parms_take_their_defaults_and_are_logged_once(
    scene: FakeHou, tmp_path: Path, caplog: Any
) -> None:
    """A node saved before a parm existed still cooks, validating, and says once how to get the parm."""
    # Read twice: the second read must not log again.
    node: FakeNode = FakeNode("/obj/topnet1/old_scheduler")
    with caplog.at_level(logging.WARNING):
        first: SchedulerSettings = read_scheduler_settings(node)
        second: SchedulerSettings = read_scheduler_settings(node)
    assert first == second == SchedulerSettings(working_dir=str(tmp_path / "scene"))
    missing: List[str] = [r.getMessage() for r in caplog.records if "has no" in r.getMessage()]
    assert len(missing) == 3
    for parm in ("'working_dir'", "'debug_level'", "'validate_graph'"):
        assert any(parm in m for m in missing), parm


@pytest.mark.parametrize("value, expected", [(0, False), (1, True)])
def test_the_validate_graph_toggle_is_read(scene: FakeHou, value: int, expected: bool) -> None:
    """The toggle is the artist's only way to cook a graph validation refuses."""
    node: FakeNode = _settings_node(working_dir="/proj", validate_graph=value)
    assert read_scheduler_settings(node).validate_graph is expected


@pytest.mark.parametrize("value", [2, -1, "high", None])
def test_a_debug_level_other_than_zero_or_one_names_the_parm(scene: FakeHou, value: Any) -> None:
    """The node offers two levels only, so any other value is a damaged scene."""
    with pytest.raises(CookSetupError, match="'debug_level'.*0 or 1"):
        read_scheduler_settings(_settings_node(working_dir="/proj", debug_level=value))


def test_the_debug_level_environment_variable_wins_when_valid(
    scene: FakeHou, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """The variable debugs a scene without editing it. An invalid value is ignored and named."""
    node: FakeNode = _settings_node(working_dir="/proj", debug_level=0)
    monkeypatch.setenv("DEADLINE_PDG_DEBUG_LEVEL", "1")
    assert read_scheduler_settings(node).debug_level == 1
    monkeypatch.setenv("DEADLINE_PDG_DEBUG_LEVEL", "2")
    with caplog.at_level(logging.WARNING):
        assert read_scheduler_settings(node).debug_level == 0
    assert "DEADLINE_PDG_DEBUG_LEVEL" in caplog.text


def test_the_host_reads_the_settings_from_its_scheduler_node(scene: FakeHou) -> None:
    """The cook reads its settings through the host, from the node PDG links to the scheduler."""
    network: Network = build_network(
        PYRO_BONFIRE, scheduler_parms={"working_dir": "/proj/out", "validate_graph": 0}
    )
    settings: SchedulerSettings = HoudiniPdgHost(network.scheduler()).read_settings()
    assert settings == SchedulerSettings(
        working_dir=os.path.abspath("/proj/out"), validate_graph=False
    )


def test_the_scheduler_node_is_found_by_name_when_pdg_gives_none(scene: FakeHou) -> None:
    """PDG doesn't always link the scheduler to its node, so the node of this type is matched by name."""
    other: FakeNode = FakeNode("/obj/topnet2/deadlinecloudscheduler2")
    mine: FakeNode = FakeNode(SCHEDULER_PATH)
    scheduler: FakePyScheduler = FakePyScheduler()
    scene.instances = [other, mine]
    assert find_scheduler_node(scheduler) is mine
    # The only node of the type is used even when its name differs.
    scene.instances = [other]
    assert find_scheduler_node(scheduler) is other
    scene.instances = []
    with pytest.raises(CookSetupError, match="deadlinecloudscheduler1"):
        find_scheduler_node(scheduler)
    # A session that can't list node types gives the same error, not a crash.
    scene.type_error = RuntimeError("no such category")
    with pytest.raises(CookSetupError, match="deadlinecloudscheduler1"):
        find_scheduler_node(scheduler)


def test_the_houdini_version_and_queue_values_come_from_the_scheduler_node(
    scene: FakeHou,
) -> None:
    """The Conda packages are pinned to the running Houdini, and the queue values are the node's own."""
    # A queue parameter is a spare parm with the shared prefix, as the ROP makes it.
    network: Network = build_network(PYRO_BONFIRE)
    spare: Any = network.scheduler_node.set_parm(
        "queue_env_do_not_use_CondaChannels", "deadline-cloud"
    )
    network.scheduler_node.spare = [spare]
    host: HoudiniPdgHost = HoudiniPdgHost(network.scheduler())
    assert host.houdini_version() == "21.0.596"
    assert host.queue_parameter_values() == [{"name": "CondaChannels", "value": "deadline-cloud"}]


# -- the main thread ----------------------------------------------------------


def test_scene_reads_from_a_pdg_thread_run_on_the_main_thread(
    scene: FakeHou, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Houdini crashes when a PDG thread reads the scene while its UI runs."""
    ran_on: List[str] = []
    monkeypatch.setitem(
        sys.modules,
        "hdefereval",
        types.SimpleNamespace(executeInMainThreadWithResult=main_thread_runner(ran_on)),
    )
    host: HoudiniPdgHost = HoudiniPdgHost(build_network(PYRO_BONFIRE).scheduler())
    results: List[Any] = []
    # Every read the cook and the submitter make, from one PDG thread.
    worker: threading.Thread = threading.Thread(
        target=lambda: results.extend(
            [host.read_graph(), host.read_scene(), host.read_settings(), host.houdini_version()]
        ),
        name="pdg",
    )
    worker.start()
    worker.join()
    assert len(results) == 4
    assert len(ran_on) >= 4 and set(ran_on) == {"pdg"}


def test_without_a_ui_scene_reads_run_on_the_calling_thread(
    scene: FakeHou, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hython has no UI to race and no ``hdefereval``, so it reads directly."""
    monkeypatch.setitem(sys.modules, "hdefereval", None)
    scene.ui = False
    host: HoudiniPdgHost = HoudiniPdgHost(build_network(PYRO_BONFIRE).scheduler())
    results: List[Any] = []
    worker: threading.Thread = threading.Thread(target=lambda: results.append(host.read_graph()))
    worker.start()
    worker.join()
    assert results


# -- pre-cook checks ----------------------------------------------------------


def _verdict(network: Network, interactive: bool = False) -> Verdict:
    """The pre-cook verdict for the network's scheduler node."""
    return PreCookChecks().run(network.scheduler_node, interactive=interactive)


def _refused(verdict: Verdict) -> List[Tuple[RefusalScenario, str]]:
    """The refusals as (scenario, subject), in order."""
    return [(r.scenario, r.subject) for r in verdict.refusals]


def test_a_cooking_example_network_is_refused_before_generation(scene: FakeHou) -> None:
    """An in-process node between steps cooks on this machine, so the job would read missing files."""
    assert _refused(_verdict(build_network(COOKING_EXAMPLE))) == [
        (RefusalScenario.IN_PROCESS_NODE, "python1")
    ]


def test_with_validation_off_the_pre_cook_refusal_becomes_a_warning(scene: FakeHou) -> None:
    """The Cook button accepts the same graph the cook submits when the artist turns validation off."""
    # The refusal is kept as an advisory that says validation is off.
    network: Network = build_network(
        COOKING_EXAMPLE, scheduler_parms={"working_dir": "$HIP", "validate_graph": 0}
    )
    verdict: Verdict = _verdict(network)
    assert verdict.accepted()
    waived: List[Any] = [
        a for a in verdict.advisories if a.scenario is RefusalScenario.IN_PROCESS_NODE
    ]
    assert [a.subject for a in waived] == ["python1"]
    assert waived[0].detail.startswith("Not refused, because graph validation is off.")


def test_with_validation_off_an_unsaved_scene_is_still_refused(scene: FakeHou) -> None:
    """Validation off waives the graph checks only. Workers can't load a scene that was never saved."""
    network: Network = build_network(
        COOKING_EXAMPLE, scheduler_parms={"working_dir": "$HIP", "validate_graph": 0}
    )
    scene.hip_file.new = True
    assert [r.scenario for r in _verdict(network).refusals] == [RefusalScenario.UNSAVED_SCENE]


def test_a_network_routed_through_a_merge_is_accepted_before_generation(scene: FakeHou) -> None:
    """A Merge between two steps cooks nothing, so it is not refused."""
    assert _verdict(build_network(ROUTED_EXAMPLE)).accepted()


def test_a_pyro_bonfire_network_is_accepted_and_no_step_is_missing_items(scene: FakeHou) -> None:
    """Before generation no step has items, so the check that needs them must not run."""
    verdict: Verdict = _verdict(build_network(PYRO_BONFIRE))
    assert verdict.accepted()
    scenarios: List[RefusalScenario] = [f.scenario for f in verdict.advisories]
    assert RefusalScenario.STEP_WITHOUT_ITEMS not in scenarios


def test_a_dynamic_node_and_a_foreign_scheduler_are_refused_before_generation(
    scene: FakeHou,
) -> None:
    """Both are known from the topology, so the artist hears before anything is uploaded."""
    network: Network = build_network(PYRO_BONFIRE)
    network.node("render").isDynamic = True
    network.node("montage").hom.set_parm("topscheduler", "localscheduler")
    network.node("montage").scheduler_name = "localscheduler"
    assert set(_refused(_verdict(network))) == {
        (RefusalScenario.DYNAMIC_NODE, "render"),
        (RefusalScenario.MULTIPLE_SCHEDULERS, "montage"),
    }


def test_an_override_naming_this_scheduler_is_not_refused_before_generation(
    scene: FakeHou,
) -> None:
    """A node that names this scheduler as its override uses this scheduler, so it is accepted."""
    network: Network = build_network(PYRO_BONFIRE)
    network.node("montage").hom.set_parm("topscheduler", SCHEDULER_PATH)
    assert _verdict(network).accepted()


@pytest.mark.parametrize("override", ["deadlinecloudscheduler1", "dc_pdg_name"])
def test_an_override_naming_this_scheduler_by_name_is_not_refused(
    scene: FakeHou, override: str
) -> None:
    """An override may name the scheduler node by its name or its PDG name, as well as its path."""
    network: Network = build_network(PYRO_BONFIRE)
    network.scheduler_node.pdg_name = "dc_pdg_name"
    network.node("montage").hom.set_parm("topscheduler", override)
    # PDG's own answer names no scheduler, so only the override's text decides.
    network.node("montage").scheduler_name = ""
    assert _verdict(network).accepted()


def test_python_code_that_opens_a_file_is_advised_before_generation(scene: FakeHou) -> None:
    """Code is too open to judge, so a node between steps whose code opens a file gets a warning."""
    network: Network = build_network(ROUTED_EXAMPLE)
    network.node("merge").hom.set_parm(
        "pythoncode", "open(work_item.inputFiles[0].path)", tags={"editorlang": "python"}
    )
    verdict: Verdict = _verdict(network)
    assert verdict.accepted()
    assert [(a.scenario, a.subject) for a in verdict.advisories] == [
        (RefusalScenario.IN_PROCESS_NODE, "merge")
    ]


def test_a_step_set_to_in_process_is_refused_before_generation(scene: FakeHou) -> None:
    """PDG never offers such a step's items, so its tasks would never run, and it is named first."""
    network: Network = build_network(PYRO_BONFIRE)
    network.node("render").cookType = 1
    assert _refused(_verdict(network)) == [(RefusalScenario.IN_PROCESS_STEP, "render")]


def test_a_non_step_writing_a_file_a_step_reads_is_refused_before_generation(
    scene: FakeHou,
) -> None:
    """A node between steps that writes a file on this machine never delivers it to the step that reads it."""
    network: Network = build_network(
        [
            ("sim", "hython sim.py", "Processor", []),
            ("textoutput1", "", "Processor", ["sim"]),
            ("render", "hython render.py", "Processor", ["textoutput1"]),
        ]
    )
    # Text Output's ``filepath`` is a write, and the render step reads the same file.
    writer: FakeNode = network.node("textoutput1").hom
    writer._type_name = "textoutput"
    writer.set_parm("filepath", "$HIP/mid/notes.txt", kind="FileReference")
    network.node("render").hom.set_parm("file", "$HIP/mid/notes.txt", kind="FileReference")
    assert (RefusalScenario.IN_PROCESS_NODE, "textoutput1") in _refused(_verdict(network))


@pytest.mark.parametrize(
    "new, unsaved, words",
    [(True, False, "never been saved"), (False, True, "unsaved changes"), (True, True, "never")],
)
def test_an_unsaved_or_new_scene_is_refused(
    scene: FakeHou, new: bool, unsaved: bool, words: str
) -> None:
    """Workers load the saved file, so they would cook a different scene than the snapshot."""
    scene.hip_file.new, scene.hip_file.unsaved = new, unsaved
    (refusal,) = _verdict(build_network(PYRO_BONFIRE)).refusals
    assert refusal.scenario is RefusalScenario.UNSAVED_SCENE
    assert words in refusal.detail
    assert refusal.remedy == "Save the scene, then cook again."
    assert refusal.subject.endswith("shot.hip")
    # The Cook button's path never prompts and never saves.
    assert scene.messages.calls == [] and scene.hip_file.saves == 0


def test_a_session_with_no_ui_is_not_refused_for_unsaved_changes(scene: FakeHou) -> None:
    """hython reports unsaved changes even right after a save, and no artist is editing the scene."""
    scene.hip_file.unsaved = True
    scene.ui = False
    assert _verdict(build_network(PYRO_BONFIRE)).accepted()


def test_a_never_saved_scene_is_refused_with_no_ui(scene: FakeHou) -> None:
    """Without a saved file there is nothing for a worker to load, UI or not."""
    scene.hip_file.new = True
    scene.ui = False
    assert [r.scenario for r in _verdict(build_network(PYRO_BONFIRE)).refusals] == [
        RefusalScenario.UNSAVED_SCENE
    ]


_BUTTONS: Tuple[str, ...] = ("Save", "Always Save", "Cancel")


def _auto_save_network(auto_save: int) -> Network:
    """A Pyro Bonfire network whose scheduler node has the auto-save toggle set to ``auto_save``."""
    return build_network(
        PYRO_BONFIRE,
        scheduler_parms={"working_dir": "$HIP", "validate_graph": 1, "auto_save_hip": auto_save},
    )


def test_save_at_the_prompt_saves_and_leaves_nothing_to_refuse(scene: FakeHou) -> None:
    """The prompt comes before the checks, so a save there clears the unsaved-scene refusal."""
    scene.hip_file.unsaved = True
    network: Network = _auto_save_network(0)
    assert _verdict(network, interactive=True).accepted()
    assert scene.hip_file.saves == 1
    assert [buttons for _, buttons in scene.messages.calls] == [_BUTTONS]
    assert network.scheduler_node.parm("auto_save_hip").sets == []  # type: ignore[union-attr]


def test_always_save_saves_and_turns_auto_save_on_so_the_next_submit_does_not_ask(
    scene: FakeHou,
) -> None:
    """The artist answers once. The toggle is the one parm written, because its value changed."""
    scene.hip_file.unsaved = True
    scene.messages.choice = 1
    network: Network = _auto_save_network(0)
    assert _verdict(network, interactive=True).accepted()
    assert scene.hip_file.saves == 1
    assert network.scheduler_node.parm("auto_save_hip").sets == [1]  # type: ignore[union-attr]
    # The next submit with changes saves without asking.
    scene.hip_file.unsaved = True
    assert _verdict(network, interactive=True).accepted()
    assert len(scene.messages.calls) == 1 and scene.hip_file.saves == 2


def test_cancel_at_the_save_prompt_raises_user_initiated_cancel(scene: FakeHou) -> None:
    """Cancel starts nothing and saves nothing, and closing the prompt is a cancel too."""
    for choice in (2, None):
        scene.messages.choice = choice
        scene.hip_file.unsaved = True
        network: Network = _auto_save_network(0)
        with pytest.raises(UserInitiatedCancel):
            _verdict(network, interactive=True)
        assert scene.hip_file.saves == 0
        assert network.scheduler_node.parm("auto_save_hip").sets == []  # type: ignore[union-attr]


def test_with_auto_save_on_unsaved_changes_are_saved_without_asking(scene: FakeHou) -> None:
    """The toggle means the artist chose to save on every submit."""
    scene.hip_file.unsaved = True
    assert _verdict(_auto_save_network(1), interactive=True).accepted()
    assert scene.messages.calls == [] and scene.hip_file.saves == 1


def test_a_saved_scene_gets_no_save_prompt(scene: FakeHou) -> None:
    """A scene with nothing to save must not ask, so a saved scene submits without that dialog."""
    assert _verdict(_auto_save_network(0), interactive=True).accepted()
    assert scene.messages.calls == [] and scene.hip_file.saves == 0


def test_a_never_saved_scene_is_refused_without_a_prompt(scene: FakeHou) -> None:
    """A new scene needs a file name, which the save prompt cannot ask for."""
    scene.hip_file.new = True
    scene.hip_file.unsaved = True
    verdict: Verdict = _verdict(build_network(PYRO_BONFIRE), interactive=True)
    assert [r.scenario for r in verdict.refusals] == [RefusalScenario.UNSAVED_SCENE]
    assert scene.messages.calls == []


def test_without_a_ui_the_interactive_check_does_not_prompt(scene: FakeHou) -> None:
    """There is no artist to answer a prompt in hython."""
    scene.ui = False
    scene.hip_file.unsaved = True
    assert _verdict(build_network(PYRO_BONFIRE), interactive=True).accepted()
    assert scene.messages.calls == []


def test_a_network_without_a_pdg_graph_creates_its_nodes_first(scene: FakeHou) -> None:
    """A network never cooked has no PDG graph, so its nodes are created, without cooking, and judged."""
    network: Network = build_network(COOKING_EXAMPLE)
    network.scheduler_node.context = None
    verdict: Verdict = _verdict(network)
    assert network.scheduler_node.cook_calls == [{"block": True, "tops_only": True}]
    assert _refused(verdict) == [(RefusalScenario.IN_PROCESS_NODE, "python1")]


def test_a_network_whose_pdg_graph_cannot_be_made_is_checked_when_it_cooks(
    scene: FakeHou, caplog: Any
) -> None:
    """Without a graph nothing can be judged yet, so the cook runs the checks when it starts."""
    network: Network = build_network(COOKING_EXAMPLE)
    network.scheduler_node.context = None
    network.scheduler_node.cook_creates = None
    with caplog.at_level(logging.WARNING):
        verdict: Verdict = _verdict(network)
    assert verdict.accepted() and "no PDG graph yet" in caplog.text


def test_a_working_directory_setting_problem_does_not_hide_the_topology(scene: FakeHou) -> None:
    """The cook reports the setting when it starts, so the checks still judge the graph."""
    network: Network = build_network(COOKING_EXAMPLE, scheduler_parms={"working_dir": ""})
    assert [r.scenario for r in _verdict(network).refusals] == [RefusalScenario.IN_PROCESS_NODE]


def test_the_scene_directory_comes_from_the_hip_when_the_environment_lacks_it(
    scene: FakeHou, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The checks resolve ``$HIP`` paths, and a session may not have ``HIP`` set."""
    monkeypatch.delenv("HIP")
    assert host_module._environment()["HIP"] == str(tmp_path / "scene")
