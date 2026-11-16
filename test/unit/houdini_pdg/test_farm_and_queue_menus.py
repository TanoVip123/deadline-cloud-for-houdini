# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The scheduler node's Farm and Queue menus, against fakes of deadline-cloud and ``hou``."""

from __future__ import annotations

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


class Menus:
    """The menus under test, with the fakes they read and the background jobs they started."""

    def __init__(self, settings: Dict[str, str]) -> None:
        self.api: FakeApi = FakeApi()
        self.config: FakeConfig = FakeConfig(settings)
        # The jobs stand in for the threads, and the deferred ones for Houdini's main thread.
        self.jobs: List[Callable[[], None]] = []
        self.deferred: List[Callable[[], None]] = []
        self.menus: FarmAndQueueMenus = FarmAndQueueMenus(
            api=self.api,
            config=self.config,
            start_thread=self.jobs.append,
            run_later_on_main_thread=self.deferred.append,
        )

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
    assert menus.menus.list_menu_items("farm") == ["farm-a", "alpha", "farm-b", "Beta"]
    assert menus.menus.list_menu_items("queue") == ["queue-y", "Comp", "queue-z", "Sims"]
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
    assert menus.menus.list_menu_items("farm") == ["farm-1", "Main", "farm-2", "Main"]


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

    menus: FarmAndQueueMenus = FarmAndQueueMenus(api=RecordingApi(), config=FakeConfig(_SAVED))
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
    assert menus.menus.list_menu_items("queue") == ["queue-y", "Comp", "queue-z", "Sims"]


_NONE_SELECTED: str = "<none selected>"


def _shown_queue(menus: Menus, node: FakeNode) -> str:
    """The Queue label Houdini shows: the label of the entry whose token the parm holds, else the value."""
    # Houdini's menu list alternates tokens and labels.
    items: List[str] = menus.menus.list_menu_items("queue")
    labels: Dict[str, str] = dict(zip(items[::2], items[1::2]))
    value: str = node.parm("queue").evalAsString()
    return labels.get(value, value)


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
    "saved_queue, shown", [("", _NONE_SELECTED), ("queue-y", "queue-y")], ids=["none", "listed"]
)
def test_a_loaded_queue_list_sets_the_queue_on_the_main_thread_only(
    scene: FakeHou, saved_queue: str, shown: str
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
