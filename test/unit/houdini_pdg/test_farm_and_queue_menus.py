# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The scheduler node's Farm and Queue menus, against fakes of deadline-cloud and ``hou``."""

from __future__ import annotations

import logging
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

# The fakes come first, because they install the fake ``hou`` the host module imports.
from .houdini_fakes import FakeHou, install_hou

from deadline.houdini_pdg.houdini.host import FarmAndQueueMenus

_LOADING: str = "Loading..."
_NONE_SELECTED: str = "<none selected>"
_LOG_IN: str = "Log in with Deadline Cloud Monitor or `deadline auth login`"


class FakeAuthenticationStatus:
    """deadline-cloud's ``AwsAuthenticationStatus``, whose values a test compares by name."""

    CONFIGURATION_ERROR: str = "CONFIGURATION_ERROR"
    AUTHENTICATED: str = "AUTHENTICATED"
    NEEDS_LOGIN: str = "NEEDS_LOGIN"


class FakeApi:
    """deadline-cloud's ``api`` module: the login check and the two list calls, each recorded."""

    AwsAuthenticationStatus: Any = FakeAuthenticationStatus

    def __init__(self) -> None:
        # A working login, two farms in two regions, and their queues, all unsorted.
        self.status: str = FakeAuthenticationStatus.AUTHENTICATED
        self.farms: List[Dict[str, str]] = [
            {"farmId": "farm-b", "displayName": "Beta", "region": "us-west-2"},
            {"farmId": "farm-a", "displayName": "alpha", "region": "us-east-1"},
        ]
        self.queues: Dict[str, List[Dict[str, str]]] = {
            "farm-a": [{"queueId": "queue-a", "displayName": "Renders"}],
            "farm-b": [
                {"queueId": "queue-z", "displayName": "Sims"},
                {"queueId": "queue-y", "displayName": "Comp"},
            ],
        }
        self.error: Optional[Exception] = None
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def check_authentication_status(self) -> str:
        self.calls.append(("check_authentication_status", {}))
        return self.status

    def list_farms(self) -> Dict[str, Any]:
        self.calls.append(("list_farms", {}))
        if self.error is not None:
            raise self.error
        return {"farms": list(self.farms)}

    def list_queues(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("list_queues", kwargs))
        return {"queues": list(self.queues[kwargs["farmId"]])}


class FakeConfig:
    """deadline-cloud's ``config`` module: settings read from a dict, and each write recorded."""

    def __init__(self, settings: Dict[str, str]) -> None:
        self.settings: Dict[str, str] = dict(settings)
        self.writes: List[Tuple[str, str]] = []

    def get_setting(self, name: str) -> str:
        return self.settings.get(name, "")

    def set_setting(self, name: str, value: str) -> None:
        self.writes.append((name, value))
        self.settings[name] = value


class FakeParm:
    """A string ``hou.Parm`` with a menu, which holds the picked token."""

    def __init__(self, value: str) -> None:
        self.value: str = value
        # Each value set, since Houdini may mark the scene unsaved on any set.
        self.sets: List[str] = []

    def evalAsString(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.sets.append(value)
        self.value = value


class FakeNode:
    """A scheduler node with its Farm and Queue menu parms."""

    def __init__(self, farm: str, queue: str) -> None:
        self.parms: Dict[str, FakeParm] = {"farm": FakeParm(farm), "queue": FakeParm(queue)}

    def parm(self, name: str) -> FakeParm:
        return self.parms[name]

    def type(self) -> Any:
        return types.SimpleNamespace(name=lambda: "deadlinecloudscheduler")

    def path(self) -> str:
        return "/obj/topnet1/scheduler1"

    def parent(self) -> Any:
        return _NETWORK


#: The TOP network that holds every scheduler node in these tests.
_NETWORK: Any = types.SimpleNamespace(path=lambda: "/obj/topnet1")


class FakePaneTab:
    """A ``hou.PaneTab`` of one type, showing one node, that records each node it's pointed at.

    Each pointing is recorded with its ``pick_node``, since True would also select the node.
    """

    def __init__(self, pane_type: str, node: Any, name: str = "panetab1") -> None:
        self.pane_type: str = pane_type
        self.node: Any = node
        self.pane_name: str = name
        self.pointed_at: List[Tuple[Any, bool]] = []

    def name(self) -> str:
        return self.pane_name

    def type(self) -> str:
        return self.pane_type

    def currentNode(self) -> Any:
        return self.node

    def setCurrentNode(self, node: Any, pick_node: bool = True) -> None:
        self.pointed_at.append((node, pick_node))
        self.node = node


class Menus:
    """The menus under test, with the fakes they read and the background jobs they started."""

    def __init__(self, settings: Dict[str, str]) -> None:
        self.api: FakeApi = FakeApi()
        self.config: FakeConfig = FakeConfig(settings)
        # The jobs stand in for the threads, and the deferred ones for Houdini's main thread.
        self.jobs: List[Callable[[], None]] = []
        self.deferred: List[Callable[[], None]] = []
        # Each redraw records the Queue values of the watched nodes, to show what was set before it.
        self.watched: List[FakeNode] = []
        self.redraws: List[List[str]] = []
        self.menus: FarmAndQueueMenus = FarmAndQueueMenus(
            api=self.api,
            config=self.config,
            start_thread=self.jobs.append,
            run_later_on_main_thread=self.deferred.append,
            redraw_menus=self.record_redraw,
        )

    def record_redraw(self) -> None:
        """Record a redraw of the menus, with the Queue value each watched node had then."""
        self.redraws.append([node.parm("queue").evalAsString() for node in self.watched])

    def run_jobs(self) -> None:
        """Run the started jobs, as their threads would."""
        # A job may start no other job, so the list is copied once.
        jobs: List[Callable[[], None]] = list(self.jobs)
        self.jobs.clear()
        for job in jobs:
            job()

    def run_deferred(self) -> None:
        """Run what the jobs queued for Houdini's main thread, in the order they queued it."""
        deferred: List[Callable[[], None]] = list(self.deferred)
        self.deferred.clear()
        for job in deferred:
            job()


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeHou:
    """A Houdini session with a UI."""
    hip: str = os.path.join(str(tmp_path), "scene", "shot.hip")
    os.makedirs(os.path.dirname(hip))
    return install_hou(monkeypatch, hip)


_SAVED: Dict[str, str] = {"defaults.farm_id": "farm-a", "defaults.queue_id": "queue-a"}


@pytest.mark.parametrize("parm", ["farm", "queue"])
def test_an_empty_list_shows_loading_starts_one_fetch_and_calls_nothing(
    scene: FakeHou, parm: str
) -> None:
    """A menu runs on Houdini's main thread, so an AWS call there would freeze the UI."""
    menus: Menus = Menus(_SAVED)
    # The one entry keeps the saved ID as its token, so the stored value doesn't change.
    saved: str = _SAVED[f"defaults.{parm}_id"]
    assert menus.menus.list_menu_items(parm) == [saved, _LOADING]
    assert menus.api.calls == []
    assert len(menus.jobs) == 1


def test_two_menu_calls_start_one_fetch_per_list(scene: FakeHou) -> None:
    """Houdini runs a menu script on every redraw, so each one must not start another fetch."""
    menus: Menus = Menus(_SAVED)
    for _ in range(2):
        menus.menus.list_menu_items("farm")
        menus.menus.list_menu_items("queue")
    assert len(menus.jobs) == 2
    # Once a list is loaded, the menu reads it and starts nothing.
    menus.run_jobs()
    menus.menus.list_menu_items("farm")
    menus.menus.list_menu_items("queue")
    assert menus.jobs == []


def test_the_fetch_fills_the_lists_and_the_menus_list_names_with_id_tokens(
    scene: FakeHou,
) -> None:
    """Houdini stores a menu's token, so the token is the ID and the label is the name."""
    menus: Menus = Menus({"defaults.farm_id": "farm-b", "defaults.queue_id": "queue-z"})
    menus.menus.list_menu_items("farm")
    menus.run_jobs()
    menus.menus.list_menu_items("queue")
    menus.run_jobs()
    # The names are sorted without case, and the queues are read in the farm's own region.
    assert menus.menus.list_menu_items("farm") == [
        "",
        _NONE_SELECTED,
        "farm-a",
        "alpha",
        "farm-b",
        "Beta",
    ]
    assert menus.menus.list_menu_items("queue") == [
        "",
        _NONE_SELECTED,
        "queue-y",
        "Comp",
        "queue-z",
        "Sims",
    ]
    assert ("list_queues", {"farmId": "farm-b", "region": "us-west-2"}) in menus.api.calls


def test_farms_with_the_same_name_are_listed_in_id_order(scene: FakeHou) -> None:
    """Two farms can share a name, and their order must not change from one load to the next."""
    # Two farms named alike, given in reverse ID order.
    menus: Menus = Menus(_SAVED)
    menus.api.farms = [
        {"farmId": "farm-2", "displayName": "Main", "region": "us-west-2"},
        {"farmId": "farm-1", "displayName": "Main", "region": "us-east-1"},
    ]
    menus.menus.list_menu_items("farm")
    menus.run_jobs()
    assert menus.menus.list_menu_items("farm")[2:] == ["farm-1", "Main", "farm-2", "Main"]


def test_the_default_fetch_runs_in_a_background_daemon_thread(scene: FakeHou) -> None:
    """A fetch on the main thread freezes Houdini, and a non-daemon one can stop it from exiting."""
    # The fake login check records the thread it runs in, then the test waits for it.
    ran: threading.Event = threading.Event()
    threads: List[threading.Thread] = []

    class RecordingApi(FakeApi):
        def check_authentication_status(self) -> str:
            threads.append(threading.current_thread())
            ran.set()
            return FakeAuthenticationStatus.NEEDS_LOGIN

    # The main-thread jobs are kept, since a test has no hdefereval.
    deferred: List[Callable[[], None]] = []
    menus: FarmAndQueueMenus = FarmAndQueueMenus(
        api=RecordingApi(), config=FakeConfig(_SAVED), run_later_on_main_thread=deferred.append
    )
    menus.list_menu_items("farm")
    assert ran.wait(timeout=10)
    assert threads[0] is not threading.main_thread() and threads[0].daemon


@pytest.mark.parametrize(
    "status", [FakeAuthenticationStatus.NEEDS_LOGIN, FakeAuthenticationStatus.CONFIGURATION_ERROR]
)
@pytest.mark.parametrize("parm", ["farm", "queue"])
def test_logged_out_each_menu_shows_the_login_entry(scene: FakeHou, parm: str, status: str) -> None:
    """The node never opens the login dialog, so the menu names the two ways to log in."""
    # Load the list with no working login, then read the menu again.
    menus: Menus = Menus(_SAVED)
    menus.api.status = status
    menus.menus.list_menu_items(parm)
    menus.run_jobs()
    # The one entry keeps the saved ID, and no list call was made without a login.
    saved: str = _SAVED[f"defaults.{parm}_id"]
    assert menus.menus.list_menu_items(parm) == [saved, _LOG_IN]
    assert [name for name, _ in menus.api.calls] == ["check_authentication_status"]


def test_a_failed_fetch_shows_its_error_in_the_menu(scene: FakeHou) -> None:
    """A list that fails in the background has nowhere else to show its error."""
    # The farm list fails, then the menu shows one entry naming the error.
    menus: Menus = Menus(_SAVED)
    menus.api.error = RuntimeError("throttled")
    menus.menus.list_menu_items("farm")
    menus.run_jobs()
    entry: List[str] = menus.menus.list_menu_items("farm")
    assert entry[0] == "farm-a" and len(entry) == 2
    assert "throttled" in entry[1]


def test_with_no_farm_the_queue_menu_asks_for_one_and_fetches_nothing(scene: FakeHou) -> None:
    """Queues are listed per farm, so with no farm there is nothing to list."""
    menus: Menus = Menus({})
    assert menus.menus.list_menu_items("queue") == ["", "Choose a farm first"]
    assert menus.jobs == []


def _loaded(settings: Dict[str, str]) -> Menus:
    """Menus with the farm list and the saved farm's queue list loaded."""
    # Each menu starts its list, the jobs fill them, and the main thread takes what they queued.
    menus: Menus = Menus(settings)
    menus.menus.list_menu_items("farm")
    menus.menus.list_menu_items("queue")
    menus.run_jobs()
    menus.run_deferred()
    return menus


def test_a_farm_pick_saves_as_select_farm_does_clears_its_queues_then_updates(
    scene: FakeHou,
) -> None:
    """deadline-cloud's select_farm clears the old farm's queue and storage profile, in this order."""
    menus: Menus = _loaded(_SAVED)
    # Load farm-b's queues too, so the pick has a list to clear.
    menus.config.settings["defaults.farm_id"] = "farm-b"
    menus.menus.list_menu_items("queue")
    menus.run_jobs()
    # Pick farm-b, recording what was saved when the update runs.
    menus.config.settings["defaults.farm_id"] = "farm-a"
    node: FakeNode = FakeNode("farm-b", "queue-a")
    saved_before_update: List[List[Tuple[str, str]]] = []
    menus.menus.save_pick(
        node, "farm", lambda: saved_before_update.append(list(menus.config.writes))
    )
    assert menus.config.writes == [
        ("defaults.farm_id", "farm-b"),
        ("defaults.farm_region", "us-west-2"),
        ("defaults.queue_id", ""),
        ("settings.storage_profile_id", ""),
    ]
    # The update ran once, after every write, and the new farm's queues load again.
    assert saved_before_update == [menus.config.writes]
    assert menus.menus.list_menu_items("queue") == ["", _LOADING]
    assert len(menus.jobs) == 1


def test_a_queue_pick_saves_as_select_queue_does_then_updates(scene: FakeHou) -> None:
    """deadline-cloud's select_queue clears the old queue's storage profile after saving the queue."""
    # Pick queue-y, recording what was saved when the update runs.
    menus: Menus = _loaded({"defaults.farm_id": "farm-b", "defaults.queue_id": "queue-z"})
    node: FakeNode = FakeNode("farm-b", "queue-y")
    updates: List[List[Tuple[str, str]]] = []
    menus.menus.save_pick(node, "queue", lambda: updates.append(list(menus.config.writes)))
    expected: List[Tuple[str, str]] = [
        ("defaults.queue_id", "queue-y"),
        ("settings.storage_profile_id", ""),
    ]
    assert menus.config.writes == expected
    assert updates == [expected]


@pytest.mark.parametrize("logged_in", [True, False], ids=["loading", "login entry"])
@pytest.mark.parametrize("parm", ["farm", "queue"])
def test_picking_a_loading_or_login_entry_does_nothing(
    scene: FakeHou, parm: str, logged_in: bool
) -> None:
    """Those entries aren't a farm or queue, so a pick must not save them or run the update."""
    # Show the Loading entry, or load the list without a login to show the login entry.
    menus: Menus = Menus(_SAVED)
    menus.api.status = (
        FakeAuthenticationStatus.AUTHENTICATED
        if logged_in
        else FakeAuthenticationStatus.NEEDS_LOGIN
    )
    menus.menus.list_menu_items(parm)
    if not logged_in:
        menus.run_jobs()
    # Houdini sets the parm to the entry's token, then runs the callback.
    node: FakeNode = FakeNode("farm-a", "queue-a")
    node.parm(parm).set(menus.menus.list_menu_items(parm)[0])
    # Nothing is saved or updated, and the parm holds the saved ID.
    updates: List[str] = []
    menus.menus.save_pick(node, parm, lambda: updates.append("update"))
    assert menus.config.writes == [] and updates == []
    assert node.parm(parm).evalAsString() == _SAVED[f"defaults.{parm}_id"]


def test_picking_the_saved_farm_again_keeps_its_queue(scene: FakeHou) -> None:
    """Saving the same farm again would clear its queue, as select_farm always does."""
    # Pick farm-a, which is already saved, with both lists loaded.
    menus: Menus = _loaded(_SAVED)
    updates: List[str] = []
    menus.menus.save_pick(FakeNode("farm-a", "queue-a"), "farm", lambda: updates.append("update"))
    assert menus.config.writes == [] and updates == []


def test_show_saved_ids_puts_the_ids_back_after_the_rop_wrote_names(scene: FakeHou) -> None:
    """The ROP's update writes the farm and queue names, and a menu stores IDs as its tokens."""
    # A node left as the ROP's update leaves it, then the saved IDs put back.
    menus: Menus = Menus(_SAVED)
    node: FakeNode = FakeNode("Farm One", "<refreshing>")
    menus.menus.show_saved_ids(node)
    assert node.parm("farm").evalAsString() == "farm-a"
    assert node.parm("queue").evalAsString() == "queue-a"


def test_show_saved_ids_puts_back_an_empty_queue_after_the_rop_wrote_none_selected(
    scene: FakeHou,
) -> None:
    """The ROP's none selected text is no menu token, so it would show the first queue's name."""
    # A farm pick leaves no queue saved, and the ROP's update then writes its text into Queue.
    menus: Menus = Menus({"defaults.farm_id": "farm-a"})
    node: FakeNode = FakeNode("Farm One", _NONE_SELECTED)
    menus.menus.show_saved_ids(node)
    assert node.parm("queue").evalAsString() == ""


def test_load_starts_both_fetches_and_reload_starts_them_again(scene: FakeHou) -> None:
    """Create and load start the lists early, and Update Queue Parameters reads them again."""
    # The first load starts the farm list and the saved farm's queue list.
    menus: Menus = Menus(_SAVED)
    menus.menus.load_lists()
    assert len(menus.jobs) == 2
    menus.run_jobs()
    # A plain load keeps what's loaded. A reload drops it, so a new login shows its farms.
    menus.menus.load_lists()
    assert menus.jobs == []
    menus.menus.load_lists(reload=True)
    assert len(menus.jobs) == 2
    assert menus.menus.list_menu_items("farm") == ["farm-a", _LOADING]


def test_without_a_ui_nothing_is_fetched_or_watched(scene: FakeHou) -> None:
    """hython loads the scene on the farm too, where the lists have no use."""
    # With no UI, a load and a menu start no job and add no selection callback.
    scene.ui = False
    menus: Menus = Menus(_SAVED)
    menus.menus.load_lists()
    menus.menus.list_menu_items("farm")
    assert menus.jobs == [] and scene.selection_callbacks == []
    assert menus.deferred == [] and menus.redraws == []


def _item(type_name: str) -> Any:
    """A selected network item of the given node type."""
    return types.SimpleNamespace(type=lambda: types.SimpleNamespace(name=lambda: type_name))


def test_selecting_a_scheduler_node_starts_the_fetch(scene: FakeHou) -> None:
    """Selecting the node starts its lists, so they are often ready when its parms show."""
    # Two loads and their jobs, which load both lists.
    menus: Menus = Menus(_SAVED)
    menus.menus.load_lists()
    menus.menus.load_lists()
    menus.run_jobs()
    # The selection callback is added once per session.
    assert len(scene.selection_callbacks) == 1
    # Another tool saves a new farm, so its queues aren't loaded yet.
    menus.config.settings["defaults.farm_id"] = "farm-b"
    scene.selection_callbacks[0]([_item("geo")])
    assert menus.jobs == []
    scene.selection_callbacks[0]([_item("geo"), _item("deadlinecloudscheduler")])
    assert len(menus.jobs) == 1
    menus.run_jobs()
    assert menus.menus.list_menu_items("queue")[2:] == ["queue-y", "Comp", "queue-z", "Sims"]


def _shown_queue(menus: Menus, node: FakeNode) -> str:
    """The Queue label Houdini shows: the entry whose token the parm holds, else the first entry."""
    # Houdini's menu list alternates tokens and labels, and a value that is no token shows the
    # first label, which made Queue look set to the first queue when it held none.
    items: List[str] = menus.menus.list_menu_items("queue")
    labels: Dict[str, str] = dict(zip(items[::2], items[1::2]))
    value: str = node.parm("queue").evalAsString()
    return labels.get(value, items[1])


def _pick_farm(menus: Menus, node: FakeNode, farm_id: str) -> None:
    """Pick ``farm_id`` in the node's Farm menu, with the update putting the saved IDs back."""
    node.parm("farm").set(farm_id)
    menus.menus.save_pick(node, "farm", lambda: menus.menus.show_saved_ids(node))


def test_a_farm_pick_starts_its_queue_fetch_at_once(scene: FakeHou) -> None:
    """The Queue menu showed Loading... until clicked, because nothing fetched the new farm's queues."""
    # Pick farm-b with both lists loaded, and no menu call after the pick.
    menus: Menus = _loaded(_SAVED)
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    _pick_farm(menus, node, "farm-b")
    assert len(menus.jobs) == 1
    menus.run_jobs()
    assert menus.api.calls[-1] == ("list_queues", {"farmId": "farm-b", "region": "us-west-2"})


@pytest.mark.parametrize(
    "saved_queue, shown, label",
    [("", "", _NONE_SELECTED), ("queue-y", "queue-y", "Comp"), ("queue-gone", "", _NONE_SELECTED)],
    ids=["none", "listed", "unlisted"],
)
def test_a_loaded_queue_list_sets_the_queue_on_the_main_thread_only(
    scene: FakeHou, saved_queue: str, shown: str, label: str
) -> None:
    """Houdini's parms are safe to set only on its main thread, so the fetch thread hands it over."""
    # Pick farm-b, then save a queue as another tool would, before its queues load.
    menus: Menus = _loaded(_SAVED)
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    _pick_farm(menus, node, "farm-b")
    menus.config.settings["defaults.queue_id"] = saved_queue
    # The fetch thread sets nothing itself, and the main thread then shows the saved queue.
    menus.run_jobs()
    assert node.parm("queue").evalAsString() == ""
    assert len(menus.deferred) == 1
    menus.run_deferred()
    assert node.parm("queue").evalAsString() == shown
    assert _shown_queue(menus, node) == label


def test_without_a_ui_a_farm_pick_fetches_and_defers_nothing(scene: FakeHou) -> None:
    """hython has no menus to show, and hdefereval exists only with a UI."""
    # Load the lists with a UI, then pick a farm with none.
    menus: Menus = _loaded(_SAVED)
    scene.ui = False
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    _pick_farm(menus, node, "farm-b")
    assert menus.jobs == [] and menus.deferred == []


def test_a_farm_and_back_shows_no_loading_once_the_queues_load(scene: FakeHou) -> None:
    """The artist saw Queue stuck on Loading... after picking a farm and then the first one again."""
    # Pick farm-b, let its queues load, then pick farm-a again and let its queues load.
    menus: Menus = _loaded(_SAVED)
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    for farm_id in ("farm-b", "farm-a"):
        _pick_farm(menus, node, farm_id)
        menus.run_jobs()
        menus.run_deferred()
    # select_farm cleared the queue, so Queue shows none selected, as the ROP does.
    assert _shown_queue(menus, node) == _NONE_SELECTED


def test_a_list_for_a_farm_no_longer_saved_leaves_the_queue_alone(scene: FakeHou) -> None:
    """A fetch can land after the next farm pick, and its queues aren't the ones the menu shows."""
    # Pick farm-b, then farm-a before farm-b's queues land, then save farm-a's queue.
    menus: Menus = _loaded(_SAVED)
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    _pick_farm(menus, node, "farm-b")
    _pick_farm(menus, node, "farm-a")
    menus.config.settings["defaults.queue_id"] = "queue-a"
    menus.run_jobs()
    # farm-a's list sets the queue first, and farm-b's, landing last, changes nothing.
    menus.deferred.reverse()
    menus.run_deferred()
    assert node.parm("queue").evalAsString() == "queue-a"


def test_a_login_entry_keeps_the_saved_queue_once_the_list_loads(scene: FakeHou) -> None:
    """The login entry's token is the saved queue, so none selected would hide the way to log in."""
    # Pick farm-b, log out, and save a queue before its list loads.
    menus: Menus = _loaded(_SAVED)
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    _pick_farm(menus, node, "farm-b")
    menus.api.status = FakeAuthenticationStatus.NEEDS_LOGIN
    menus.config.settings["defaults.queue_id"] = "queue-z"
    menus.run_jobs()
    menus.run_deferred()
    assert _shown_queue(menus, node) == _LOG_IN


def test_with_no_farm_the_farm_list_leaves_the_queue_alone(scene: FakeHou) -> None:
    """With no farm, Queue shows Choose a farm first, and the farm list has no queues to show."""
    # Load the farm list with no farm saved.
    menus: Menus = Menus({})
    node: FakeNode = FakeNode("", "")
    scene.instances = [node]
    menus.menus.list_menu_items("farm")
    menus.run_jobs()
    menus.run_deferred()
    assert _shown_queue(menus, node) == "Choose a farm first"


def test_a_reload_before_the_main_thread_runs_leaves_the_queue_for_the_next_list(
    scene: FakeHou,
) -> None:
    """A reload drops the list, so the queue waits for the next one instead of showing none selected."""
    # Pick farm-b, let its queues load, then reload before the main thread sets the queue.
    menus: Menus = _loaded(_SAVED)
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    _pick_farm(menus, node, "farm-b")
    menus.run_jobs()
    menus.menus.load_lists(reload=True)
    menus.run_deferred()
    assert node.parm("queue").evalAsString() == ""


class FakeDeferEval:
    """``hdefereval`` with only ``executeDeferred``, which keeps each job for the test to run."""

    def __init__(self) -> None:
        self.queued: List[Callable[[], None]] = []

    def executeDeferred(self, job: Callable[[], None]) -> None:
        self.queued.append(job)


def test_the_default_hand_off_queues_the_queue_set_for_every_node_and_returns(
    scene: FakeHou, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parm set from the fetch thread is unsafe in Houdini, and a wait on the main thread can hang it."""
    # The fake has no executeInMainThreadWithResult, so a hand-off that waits fails here.
    deferred: FakeDeferEval = FakeDeferEval()
    monkeypatch.setitem(sys.modules, "hdefereval", deferred)
    # Every scheduler node shares the saved farm and queue, so each one's Queue is set.
    nodes: List[FakeNode] = [FakeNode("farm-a", ""), FakeNode("farm-a", "")]
    scene.instances = list(nodes)
    jobs: List[Callable[[], None]] = []
    menus: FarmAndQueueMenus = FarmAndQueueMenus(
        api=FakeApi(), config=FakeConfig(_SAVED), start_thread=jobs.append
    )
    menus.list_menu_items("queue")
    for job in jobs:
        job()
    # The fetch only queued the set, and the main thread's run of it sets both nodes.
    assert [node.parm("queue").evalAsString() for node in nodes] == ["", ""]
    assert len(deferred.queued) == 1
    deferred.queued[0]()
    assert [node.parm("queue").evalAsString() for node in nodes] == ["queue-a", "queue-a"]


def test_a_queue_that_already_shows_the_saved_queue_is_not_set_again(scene: FakeHou) -> None:
    """Houdini may mark a scene unsaved on any parm set, so a scene that just opened stays saved."""
    # One node already holds the saved queue, and the other an old one.
    menus: Menus = Menus(_SAVED)
    unchanged: FakeNode = FakeNode("farm-a", "queue-a")
    changed: FakeNode = FakeNode("farm-a", "queue-old")
    scene.instances = [unchanged, changed]
    menus.menus.list_menu_items("queue")
    menus.run_jobs()
    menus.run_deferred()
    assert unchanged.parm("queue").sets == []
    assert changed.parm("queue").sets == ["queue-a"]


@pytest.mark.parametrize("parm", ["farm", "queue"])
def test_a_loaded_list_queues_one_main_thread_job_that_redraws_the_menus_once(
    scene: FakeHou, parm: str
) -> None:
    """Queue showed Loading... until clicked, since a list landing never told Houdini to redraw."""
    # The fetch thread only queues the job, and the main thread's run of it redraws once.
    menus: Menus = Menus(_SAVED)
    menus.menus.list_menu_items(parm)
    menus.run_jobs()
    assert len(menus.deferred) == 1 and menus.redraws == []
    menus.run_deferred()
    assert len(menus.redraws) == 1


def test_an_unchanged_queue_is_not_set_and_the_menus_still_redraw(scene: FakeHou) -> None:
    """A set may mark a scene unsaved, so the label must update by a redraw, not by a set."""
    # The node already holds the saved queue when its list lands.
    menus: Menus = Menus(_SAVED)
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    menus.watched = [node]
    menus.menus.list_menu_items("queue")
    menus.run_jobs()
    menus.run_deferred()
    assert node.parm("queue").sets == []
    assert menus.redraws == [["queue-a"]]


def test_a_changed_queue_is_set_before_the_menus_redraw(scene: FakeHou) -> None:
    """A redraw before the set would draw the old queue's label until the next redraw."""
    # The node holds an old queue, so the job sets the saved one, then redraws.
    menus: Menus = Menus(_SAVED)
    node: FakeNode = FakeNode("farm-a", "queue-old")
    scene.instances = [node]
    menus.watched = [node]
    menus.menus.list_menu_items("queue")
    menus.run_jobs()
    menus.run_deferred()
    assert node.parm("queue").sets == ["queue-a"]
    assert menus.redraws == [["queue-a"]]


def _default_redraw_menus(scene: FakeHou) -> Tuple[FarmAndQueueMenus, List[Callable[[], None]]]:
    """Menus with the default redraw, both lists loaded, and their main-thread jobs not yet run."""
    # The fetch jobs run here, and the main-thread jobs are returned for the test to run.
    jobs: List[Callable[[], None]] = []
    deferred: List[Callable[[], None]] = []
    menus: FarmAndQueueMenus = FarmAndQueueMenus(
        api=FakeApi(),
        config=FakeConfig(_SAVED),
        start_thread=jobs.append,
        run_later_on_main_thread=deferred.append,
    )
    menus.load_lists()
    for job in jobs:
        job()
    return menus, deferred


def test_the_default_redraw_points_a_parameters_pane_at_the_network_then_back_at_its_node(
    scene: FakeHou,
) -> None:
    """Pointing a Parameters pane at the node it already shows didn't redraw the menu labels.

    Pointing it away and back rebuilds the pane. ``pick_node=False`` keeps the selection, and
    no parm is set, since a set could mark the scene unsaved.
    """
    # The node already shows the saved queue, so nothing but the redraws should happen.
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    pane: FakePaneTab = FakePaneTab("Parm", node)
    scene.pane_tabs = [pane]
    _, deferred = _default_redraw_menus(scene)
    for job in deferred:
        job()
    # One redraw for each of the two lists, each away to the network and back to the node.
    away_and_back: List[Tuple[Any, bool]] = [(_NETWORK, False), (node, False)]
    assert pane.pointed_at == away_and_back * 2 and pane.currentNode() is node
    assert node.parm("farm").sets == [] and node.parm("queue").sets == []


@pytest.mark.parametrize(
    "pane_type, shown",
    [("Parm", _item("geo")), ("Parm", None), ("NetworkEditor", FakeNode("farm-a", "queue-a"))],
    ids=["another node", "no node", "not a parameters pane"],
)
def test_the_default_redraw_leaves_other_panes_alone(
    scene: FakeHou, pane_type: str, shown: Any
) -> None:
    """Re-pointing any other pane would reset what the artist was looking at in it."""
    # A pane showing another node, or nothing, or a network, next to the lists loading.
    pane: FakePaneTab = FakePaneTab(pane_type, shown)
    scene.pane_tabs = [pane]
    _, deferred = _default_redraw_menus(scene)
    for job in deferred:
        job()
    assert pane.pointed_at == [] and pane.currentNode() is shown


def test_the_default_redraw_does_nothing_without_a_ui(scene: FakeHou) -> None:
    """hython has no menus to redraw, and its hou.ui may be missing."""
    # The lists load with a UI, and the UI is gone when the main-thread jobs run.
    pane: FakePaneTab = FakePaneTab("Parm", FakeNode("farm-a", "queue-a"))
    scene.pane_tabs = [pane]
    _, deferred = _default_redraw_menus(scene)
    scene.ui = False
    for job in deferred:
        job()
    assert pane.pointed_at == []


@pytest.mark.parametrize("parm", ["farm", "queue"])
def test_a_loaded_menu_starts_with_a_none_selected_entry_whose_token_is_empty(
    scene: FakeHou, parm: str
) -> None:
    """With no entry for an empty value, Houdini showed the first farm or queue as if it were set."""
    menus: Menus = _loaded(_SAVED)
    assert menus.menus.list_menu_items(parm)[:2] == ["", _NONE_SELECTED]


def test_a_farm_with_no_queues_lists_only_the_none_selected_entry(scene: FakeHou) -> None:
    """A farm with no queues still needs an entry the empty saved queue matches."""
    # farm-a lists no queues, and its empty list loads.
    menus: Menus = Menus({"defaults.farm_id": "farm-a"})
    menus.api.queues["farm-a"] = []
    menus.menus.list_menu_items("queue")
    menus.run_jobs()
    assert menus.menus.list_menu_items("queue") == ["", _NONE_SELECTED]


@pytest.mark.parametrize("farms", [["farm-b"], ["farm-b", "farm-a"]], ids=["farm", "and back"])
def test_after_a_farm_pick_and_its_queues_landing_the_queue_is_empty(
    scene: FakeHou, farms: List[str]
) -> None:
    """The ROP's none selected text is no menu token, so Queue showed the first queue's name."""
    # Pick each farm, and let its queues load and reach the main thread.
    menus: Menus = _loaded(_SAVED)
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    for farm_id in farms:
        _pick_farm(menus, node, farm_id)
        menus.run_jobs()
        menus.run_deferred()
    assert node.parm("queue").evalAsString() == ""
    assert _shown_queue(menus, node) == _NONE_SELECTED


@pytest.mark.parametrize(
    "farms", [["farm-b"], ["farm-b", "farm-a"], ["farm-a", "farm-b"]], ids=["b", "b, a", "a, b"]
)
def test_a_farm_pick_never_saves_or_shows_a_listed_queue(scene: FakeHou, farms: List[str]) -> None:
    """A farm pick clears the queue, so a listed queue in the settings or in Queue would be wrong."""
    # Pick the farms in turn, with every queue list landing, and the saved queue listed at first.
    menus: Menus = _loaded(_SAVED)
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    for farm_id in farms:
        _pick_farm(menus, node, farm_id)
        menus.run_jobs()
        menus.run_deferred()
    listed: List[str] = [
        queue["queueId"] for queues in menus.api.queues.values() for queue in queues
    ]
    saved_queues: List[str] = [
        value for name, value in menus.config.writes if name == "defaults.queue_id"
    ]
    assert saved_queues and not set(saved_queues) & set(listed)
    assert not set(node.parm("queue").sets) & set(listed)


@pytest.mark.parametrize("saved_queue", ["queue-a", ""], ids=["queue saved", "no queue saved"])
@pytest.mark.parametrize("parm", ["farm", "queue"])
def test_picking_none_selected_saves_nothing(scene: FakeHou, parm: str, saved_queue: str) -> None:
    """The entry is no farm or queue, so a pick of it must not clear or change the saved one."""
    # Houdini sets the parm to the entry's empty token, then runs the callback.
    settings: Dict[str, str] = {"defaults.farm_id": "farm-a", "defaults.queue_id": saved_queue}
    menus: Menus = _loaded(settings)
    node: FakeNode = FakeNode("farm-a", saved_queue)
    node.parm(parm).set("")
    updates: List[str] = []
    menus.menus.save_pick(node, parm, lambda: updates.append("update"))
    # Nothing is saved or updated, and the parm holds the saved ID again.
    assert menus.config.writes == [] and updates == []
    assert node.parm(parm).evalAsString() == settings[f"defaults.{parm}_id"]


# -- the menus' INFO log lines ---------------------------------------------------

#: The package logger the artist sets to INFO in Houdini's Python Shell to see these lines.
_PACKAGE_LOGGER: str = "deadline.houdini_pdg"
_PREFIX: str = "Farm and Queue menus: "


def _menu_lines(caplog: pytest.LogCaptureFixture, part: str = "") -> List[str]:
    """The menus' INFO lines, without their prefix, that contain ``part``."""
    # Only INFO lines from the host module's logger count, as the console would show them.
    lines: List[str] = [
        record.getMessage()[len(_PREFIX) :]
        for record in caplog.records
        if record.name == "deadline.houdini_pdg.houdini.host"
        and record.levelno == logging.INFO
        and record.getMessage().startswith(_PREFIX)
    ]
    return [line for line in lines if part in line]


def test_starting_and_landing_a_fetch_log_the_list_and_its_count(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """The artist's console must show whether each fetch started and landed, to find where it stops.

    Setting only the package logger to INFO must be enough, as in the Python Shell.
    """
    # Both lists start and land, with the package logger at INFO and nothing else set.
    caplog.set_level(logging.INFO, logger=_PACKAGE_LOGGER)
    menus: Menus = Menus(_SAVED)
    menus.menus.load_lists()
    menus.run_jobs()
    assert _menu_lines(caplog, "loading the") == [
        "started loading the farms.",
        "started loading the queues of farm farm-a.",
    ]
    assert _menu_lines(caplog, "loaded the") == [
        "loaded the farms: 2 item(s).",
        "loaded the queues of farm farm-a: 1 item(s).",
    ]


def test_a_fetch_already_loading_logs_no_second_start(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """A start line for a fetch that never ran would point the search at the wrong place."""
    # The second load finds both lists loading, so it starts nothing and logs nothing.
    caplog.set_level(logging.INFO, logger=_PACKAGE_LOGGER)
    menus: Menus = Menus(_SAVED)
    menus.menus.load_lists()
    menus.menus.load_lists()
    assert len(_menu_lines(caplog, "started loading")) == 2


@pytest.mark.parametrize(
    "logged_in, error, landed",
    [
        (False, None, "the login entry"),
        (True, RuntimeError("no network"), "the error entry (no network)"),
    ],
    ids=["login", "error"],
)
def test_a_landed_fetch_logs_the_login_or_error_entry(
    scene: FakeHou,
    caplog: pytest.LogCaptureFixture,
    logged_in: bool,
    error: Optional[Exception],
    landed: str,
) -> None:
    """A fetch that lands with one entry is no list, so the line must say which entry it shows."""
    # Log out or break the farm list, then let the farm fetch land.
    caplog.set_level(logging.INFO, logger=_PACKAGE_LOGGER)
    menus: Menus = Menus(_SAVED)
    if not logged_in:
        menus.api.status = FakeAuthenticationStatus.NEEDS_LOGIN
    menus.api.error = error
    menus.menus.list_menu_items("farm")
    menus.run_jobs()
    assert _menu_lines(caplog, "loaded the") == [f"loaded the farms: {landed}."]


def test_the_main_thread_job_logs_its_list_and_what_the_cache_holds(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """A job that runs while the cache is empty again would leave the menus on Loading..."""
    caplog.set_level(logging.INFO, logger=_PACKAGE_LOGGER)
    # Only the queue menu asks, so the farm list was never started.
    menus: Menus = Menus(_SAVED)
    scene.instances = [FakeNode("farm-a", "queue-a")]
    menus.menus.list_menu_items("queue")
    menus.run_jobs()
    menus.run_deferred()
    # Then both lists reload, and only the farm list lands before its main-thread job runs.
    menus.menus.load_lists(reload=True)
    farms_job: Callable[[], None] = menus.jobs[0]
    farms_job()
    menus.run_deferred()
    assert _menu_lines(caplog, "main thread") == [
        "the main thread shows the queues of farm farm-a. The cache holds farms missing,"
        " and queues of farm 'farm-a' loaded with 1 item(s).",
        "the main thread shows the farms. The cache holds farms loaded with 2 item(s),"
        " and queues of farm 'farm-a' loading.",
    ]


def test_showing_the_saved_queue_logs_the_value_and_the_nodes_it_set(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """The line must tell a Queue left unset apart from one set, and from a list that was skipped."""
    # Pick farm-b, save its queue-y, and let the list land: one node changes, one already shows it.
    menus: Menus = _loaded(_SAVED)
    changed: FakeNode = FakeNode("farm-a", "queue-a")
    unchanged: FakeNode = FakeNode("farm-b", "queue-y")
    scene.instances = [changed, unchanged]
    _pick_farm(menus, changed, "farm-b")
    menus.config.settings["defaults.queue_id"] = "queue-y"
    unchanged.parm("queue").set("queue-y")
    caplog.set_level(logging.INFO, logger=_PACKAGE_LOGGER)
    caplog.clear()
    menus.run_jobs()
    menus.run_deferred()
    assert _menu_lines(caplog, "saved farm") == [
        "saved farm 'farm-b', queue 'queue-y'. Queue shows 'queue-y', set on 1 node(s)."
    ]


@pytest.mark.parametrize(
    "reload, line",
    [
        (False, "The queues of farm 'farm-b' aren't the saved farm's, so no node is set."),
        (True, "The queues of farm 'farm-b' aren't loaded, so no node is set."),
    ],
    ids=["another farm", "reloaded"],
)
def test_a_skipped_queue_list_logs_why_no_node_is_set(
    scene: FakeHou, caplog: pytest.LogCaptureFixture, reload: bool, line: str
) -> None:
    """A list for another farm, or one a reload dropped, sets nothing, and the line says which."""
    # Pick farm-b and let its queues land, then pick farm-a or reload before the main thread runs.
    menus: Menus = _loaded(_SAVED)
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    _pick_farm(menus, node, "farm-b")
    menus.run_jobs()
    if reload:
        menus.menus.load_lists(reload=True)
    else:
        _pick_farm(menus, node, "farm-a")
    caplog.set_level(logging.INFO, logger=_PACKAGE_LOGGER)
    caplog.clear()
    menus.run_deferred()
    saved_farm: str = "farm-b" if reload else "farm-a"
    assert _menu_lines(caplog, "saved farm") == [f"saved farm '{saved_farm}', queue ''. {line}"]


def test_the_redraw_logs_the_panes_it_saw_and_the_one_it_re_pointed(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """The console must show whether the redraw found the artist's Parameters pane at all."""
    # A Parameters pane on the scheduler node, and a network editor, for one list landing.
    node: FakeNode = FakeNode("farm-a", "queue-a")
    scene.instances = [node]
    scene.pane_tabs = [
        FakePaneTab("Parm", node, "parameditor1"),
        FakePaneTab("NetworkEditor", node, "networkeditor1"),
    ]
    _, deferred = _default_redraw_menus(scene)
    caplog.set_level(logging.INFO, logger=_PACKAGE_LOGGER)
    caplog.clear()
    deferred[0]()
    assert _menu_lines(caplog, "pane") == [
        "the redraw sees 2 pane tabs.",
        "pane parameditor1 shows /obj/topnet1/scheduler1, so the redraw calls"
        " setCurrentNode(/obj/topnet1, pick_node=False), then"
        " setCurrentNode(/obj/topnet1/scheduler1, pick_node=False).",
        "1 of 2 pane tabs show a scheduler node.",
    ]
    # setCurrentNode returns None when it works, so that alone is not logged.
    assert _menu_lines(caplog, "returned None") == []


def test_the_redraw_logs_each_call_that_failed_or_returned_none(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """A redraw call that fails is otherwise only a debug line, so the console would show nothing."""
    # A Parameters pane with no node, one whose type fails, and one whose setCurrentNode fails.
    node: FakeNode = FakeNode("farm-a", "queue-a")
    empty: FakePaneTab = FakePaneTab("Parm", None)
    no_type: FakePaneTab = FakePaneTab("Parm", node)
    no_redraw: FakePaneTab = FakePaneTab("Parm", node)

    # A pane call that raises, as Houdini's does for a pane that was closed.
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("pane closed")

    # The failing calls replace the fakes' methods, and one list lands.
    setattr(no_type, "type", fail)
    setattr(no_redraw, "setCurrentNode", fail)
    scene.pane_tabs = [empty, no_type, no_redraw]
    _, deferred = _default_redraw_menus(scene)
    caplog.set_level(logging.DEBUG, logger=_PACKAGE_LOGGER)
    caplog.clear()
    deferred[0]()
    # Each failure also keeps its traceback at DEBUG, as every other pane call's does.
    assert any(r.levelno == logging.DEBUG and r.exc_info for r in caplog.records)
    problems: List[str] = [
        line for line in _menu_lines(caplog) if " failed: " in line or "returned None" in line
    ]
    assert problems == [
        "pane.currentNode() returned None.",
        "node.type() failed: 'NoneType' object has no attribute 'type'",
        "node.type().name() failed: 'NoneType' object has no attribute 'name'",
        "pane.type() failed: pane closed",
        "pane.setCurrentNode(parent) failed: pane closed",
        "pane.setCurrentNode(node) failed: pane closed",
    ]


def test_the_redraw_logs_when_hou_ui_cannot_list_the_panes(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """With no pane list nothing is redrawn, and the console must show why."""
    # The fake's paneTabs raises on a missing list, as a broken hou.ui would.
    _, deferred = _default_redraw_menus(scene)
    setattr(scene, "pane_tabs", None)
    caplog.set_level(logging.INFO, logger=_PACKAGE_LOGGER)
    caplog.clear()
    deferred[0]()
    assert _menu_lines(caplog, "hou.ui") == [
        "hou.ui.paneTabs() failed: 'NoneType' object is not iterable"
    ]
    assert _menu_lines(caplog, "pane tabs") == [
        "the redraw sees 0 pane tabs.",
        "0 of 0 pane tabs show a scheduler node.",
    ]


def test_the_redraw_logs_that_hython_has_nothing_to_redraw(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """A session that lost its UI redraws nothing, and the line says so instead of a pane count."""
    # The lists load with a UI, and the UI is gone when the first main-thread job runs.
    _, deferred = _default_redraw_menus(scene)
    scene.ui = False
    caplog.set_level(logging.INFO, logger=_PACKAGE_LOGGER)
    caplog.clear()
    deferred[0]()
    assert _menu_lines(caplog, "redraw") == ["no UI, so nothing to redraw."]


def _logged_menu(menus: Menus, parm: str, caplog: pytest.LogCaptureFixture) -> List[str]:
    """The menu lines that one call of ``parm``'s menu logs."""
    caplog.clear()
    menus.menus.list_menu_items(parm)
    return _menu_lines(caplog, "menu returned")


def test_each_menu_call_logs_one_line_naming_what_it_returned(
    scene: FakeHou, caplog: pytest.LogCaptureFixture
) -> None:
    """A menu stuck on Loading... while its list is loaded would show here as the placeholder."""
    caplog.set_level(logging.INFO, logger=_PACKAGE_LOGGER)
    # Before and after the lists load, and with no farm saved.
    menus: Menus = Menus(_SAVED)
    assert _logged_menu(menus, "farm", caplog) == [
        'the farm menu returned the placeholder "Loading...".'
    ]
    menus.run_jobs()
    assert _logged_menu(menus, "farm", caplog) == [
        "the farm menu returned the loaded list of 2 item(s)."
    ]
    menus.config.settings["defaults.farm_id"] = ""
    assert _logged_menu(menus, "queue", caplog) == [
        'the queue menu returned the placeholder "Choose a farm first".'
    ]


@pytest.mark.parametrize(
    "logged_in, error, shown",
    [(False, None, "the login entry"), (True, RuntimeError("no network"), "the error entry")],
    ids=["login", "error"],
)
def test_a_menu_call_logs_the_login_or_error_entry(
    scene: FakeHou,
    caplog: pytest.LogCaptureFixture,
    logged_in: bool,
    error: Optional[Exception],
    shown: str,
) -> None:
    """A loaded login or error entry is one entry, like a placeholder, and the line tells them apart."""
    # Log out or break the farm list, let it land, then call the farm menu once.
    caplog.set_level(logging.INFO, logger=_PACKAGE_LOGGER)
    menus: Menus = Menus(_SAVED)
    if not logged_in:
        menus.api.status = FakeAuthenticationStatus.NEEDS_LOGIN
    menus.api.error = error
    menus.menus.list_menu_items("farm")
    menus.run_jobs()
    assert _logged_menu(menus, "farm", caplog) == [f"the farm menu returned {shown}."]
