# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The registration module, which builds a cook's objects, against fakes of ``pdg``, ``hou`` and Deadline Cloud."""

from __future__ import annotations

import importlib.util
import logging
import os
import types
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from deadline.client.api import _telemetry
from hypothesis import HealthCheck, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from .houdini_fakes import (
    COOKING_EXAMPLE,
    PDG,
    PYRO_BONFIRE,
    FakeHou,
    Network,
    build_network,
    install_hou,
)

import deadline.houdini_pdg
from deadline.houdini_pdg.cook import CookServices, ScheduleResult, SystemClock
from deadline.houdini_pdg.errors import CookSetupError
from deadline.houdini_pdg.houdini.host import HoudiniPdgHost
from deadline.houdini_pdg.houdini.settings import SchedulerSettings
from deadline.houdini_pdg.ports import CookPhase, CookSummary

#: The registration module's source, found from the installed package. PDG loads it by path, so the test does too.
_REGISTRATION: str = os.path.join(
    os.path.dirname(deadline.houdini_pdg.__file__),
    *("houdini_plugin", "pdg", "types", "deadlinecloud", "__init__.py"),
)


def _load() -> ModuleType:
    """The registration module, loaded from its file the way PDG loads it."""
    spec: Any = importlib.util.spec_from_file_location("deadlinecloud_under_test", _REGISTRATION)
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REGISTRATION: ModuleType = _load()

#: The client library's cache of telemetry clients, by package name.
_TELEMETRY_CLIENTS: Dict[str, Any] = getattr(_telemetry, "__cached_telemetry_clients")


def _client_error(code: str) -> ClientError:
    """A botocore error with the service code ``code``."""
    return ClientError({"Error": {"Code": code, "Message": code}}, "Op")


class FakeS3:
    """An S3 client whose ``get_object`` raises ``error``, and that records the keys it was asked for."""

    def __init__(self, error: Optional[Exception]) -> None:
        self.error: Optional[Exception] = error
        self.keys: List[Tuple[str, str]] = []

    def get_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:
        self.keys.append((Bucket, Key))
        if self.error is not None:
            raise self.error
        return {"Body": b""}


class FakeDeadlineClient:
    """A Deadline Cloud client whose ``get_queue`` gives each answer in turn: a queue or an error."""

    def __init__(self, answers: List[Any]) -> None:
        self.answers: List[Any] = answers

    def get_queue(self, *, farmId: str, queueId: str) -> Dict[str, Any]:
        answer: Any = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return dict(answer)


_QUEUE: Dict[str, Any] = {
    "displayName": "Render",
    "jobAttachmentSettings": {"s3BucketName": "bucket-a", "rootPrefix": "DeadlineCloud"},
}


class FakeCloud(REGISTRATION._ClientFactory):  # type: ignore[misc, name-defined]
    """Clients with a fixed farm, queue, S3 answer, and telemetry, and no waiting."""

    def __init__(
        self,
        *,
        settings: Optional[Dict[str, str]] = None,
        queue: Optional[List[Any]] = None,
        s3_error: Optional[Exception] = _client_error("NoSuchKey"),
        login_needed: bool = False,
    ) -> None:
        self.settings: Dict[str, str] = (
            settings
            if settings is not None
            else {"defaults.farm_id": "farm-1", "defaults.queue_id": "queue-1"}
        )
        self.client: FakeDeadlineClient = FakeDeadlineClient(queue or [_QUEUE])
        self.s3: FakeS3 = FakeS3(s3_error)
        self.events: List[Tuple[str, Dict[str, Any]]] = []
        self.sleeps: List[float] = []
        self.login_needed: bool = login_needed
        self.session_names: List[str] = []

    def get_setting(self, name: str) -> str:
        return self.settings.get(name, "")

    def deadline_client(self) -> Any:
        return self.client

    def queue_session(self, client: Any, farm_id: str, queue_id: str, display_name: str) -> Any:
        self.session_names.append(display_name)
        return type("Session", (), {"client": lambda _self, name: self.s3})()

    def needs_login(self) -> bool:
        return self.login_needed

    def record_event(self) -> Callable[[str, Dict[str, Any]], Any]:
        return lambda event_type, details: self.events.append((event_type, details))

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeHou:
    """A saved scene at ``<tmp>/scene/shot.hip``, in hython, with no UI."""
    hip: str = os.path.join(str(tmp_path), "scene", "shot.hip")
    os.makedirs(os.path.dirname(hip))
    monkeypatch.delenv("DEADLINE_PDG_DEBUG_LEVEL", raising=False)
    monkeypatch.setenv("HIP", os.path.dirname(hip))
    state: FakeHou = install_hou(monkeypatch, hip)
    state.ui = False
    return state


def _scheduler(network: Network, cloud: Optional[FakeCloud] = None) -> Any:
    """The registered scheduler for ``network``, using ``cloud`` for every Deadline Cloud call."""
    chosen: FakeCloud = cloud or FakeCloud()

    class Scheduler(REGISTRATION.DeadlineCloudPDGScheduler):  # type: ignore[misc, name-defined]
        def _clients(self) -> Any:
            return chosen

    scheduler: Any = Scheduler(None, "deadlinecloudscheduler1")
    scheduler.context = network.context
    scheduler.node = network.scheduler_node
    return scheduler


def _connect(network: Network, cloud: FakeCloud, tmp_path: Path) -> CookServices:
    """The services the composition root builds for one cook."""
    host: HoudiniPdgHost = HoudiniPdgHost(_scheduler(network, cloud))
    settings: SchedulerSettings = SchedulerSettings(working_dir=str(tmp_path / "scene"))
    services: CookServices = REGISTRATION._connect(
        settings, "cook-1", host=host, clock=SystemClock(), clients=cloud
    )
    return services


# -- start, tick and stop -----------------------------------------------------


def test_a_refused_graph_fails_the_start_on_the_node_and_cancels_on_the_first_tick(
    scene: FakeHou,
) -> None:
    """PDG drops a scheduler's error when the start returns False, so the start says True and the tick cancels."""
    scheduler: Any = _scheduler(build_network(COOKING_EXAMPLE))
    assert scheduler.onStartCook(False, None) is True
    errors: List[str] = [call[1] for call in scheduler.calls if call[0] == "cookError"]
    assert len(errors) == 1 and "python1" in errors[0]
    assert scheduler.onTick() == PDG.tickResult.SchedulerCancelCook
    assert scheduler.onSchedule(object()) == PDG.scheduleResult.CookFailed


def test_a_failed_start_works_in_a_temporary_directory_that_stop_removes(scene: FakeHou) -> None:
    """Without a working directory PDG copies its job scripts to the filesystem root."""
    scheduler: Any = _scheduler(build_network(COOKING_EXAMPLE))
    scheduler.onStartCook(False, None)
    (directory,) = [call[1] for call in scheduler.calls if call[0] == "setWorkingDir"]
    assert os.path.isdir(directory) and os.path.basename(directory).startswith("deadline_pdg_")
    assert scheduler.onStopCook(False) is True
    assert not os.path.exists(directory)


def test_a_generate_sets_the_working_directory_and_skips_the_checks(
    scene: FakeHou, tmp_path: Path
) -> None:
    """A generate submits nothing, so a graph the cook would refuse still generates."""
    scheduler: Any = _scheduler(build_network(COOKING_EXAMPLE))
    assert scheduler.onStartCook(True, None) is True
    folder: str = str(tmp_path / "scene")
    assert ("setWorkingDir", folder, folder) in scheduler.calls
    assert not [call for call in scheduler.calls if call[0] == "cookError"]


@pytest.mark.parametrize(
    "cook_type", ["cookType.GenerateNode", types.SimpleNamespace(name="GenerateNode")]
)
def test_a_generate_houdini_reports_in_the_cook_options_is_not_a_farm_cook(
    scene: FakeHou, tmp_path: Path, cook_type: Any
) -> None:
    """Houdini starts every generate with static False, so a farm cook would need a farm to generate."""
    network: Network = build_network(COOKING_EXAMPLE)
    network.context.cookOptions = types.SimpleNamespace(cookType=cook_type)
    # No farm is set, so a farm cook would fail on the missing farm ID.
    scheduler: Any = _scheduler(network, FakeCloud(settings={}))
    assert scheduler.onStartCook(False, None) is True
    folder: str = str(tmp_path / "scene")
    assert ("setWorkingDir", folder, folder) in scheduler.calls
    assert not [call for call in scheduler.calls if call[0] == "cookError"]
    assert scheduler.onTick() == PDG.tickResult.SchedulerReady


def test_a_cook_houdini_reports_in_the_cook_options_is_checked_and_connected(
    scene: FakeHou,
) -> None:
    """Only a generate skips the checks, so a cook of a refused graph still fails its start."""
    # The same refused graph, with Houdini's cook options saying it's a cook.
    network: Network = build_network(COOKING_EXAMPLE)
    network.context.cookOptions = types.SimpleNamespace(cookType="cookType.Node")
    scheduler: Any = _scheduler(network)
    scheduler.onStartCook(False, None)
    errors: List[str] = [call[1] for call in scheduler.calls if call[0] == "cookError"]
    assert len(errors) == 1 and "python1" in errors[0]


def test_a_context_without_cook_options_is_a_cook(scene: FakeHou) -> None:
    """Only a generate PDG reports may skip the checks, so unreadable options mean a cook."""
    # A refused graph whose context has no cook options must still fail its start.
    network: Network = build_network(COOKING_EXAMPLE)
    network.context.cookOptions = None
    scheduler: Any = _scheduler(network)
    scheduler.onStartCook(False, None)
    assert [call for call in scheduler.calls if call[0] == "cookError"]


def test_a_cook_started_in_a_session_with_a_ui_never_opens_the_save_prompt(scene: FakeHou) -> None:
    """Only the node's Submit asks to save. A cook started any other way refuses unsaved changes."""
    # Start a cook of an unsaved scene in a session with a UI, and expect a refusal, not a prompt.
    scene.ui = True
    scene.hip_file.unsaved = True
    scheduler: Any = _scheduler(build_network(PYRO_BONFIRE))
    scheduler.onStartCook(False, None)
    assert scene.messages.calls == [] and scene.hip_file.saves == 0
    errors: List[str] = [call[1] for call in scheduler.calls if call[0] == "cookError"]
    assert len(errors) == 1 and "unsaved" in errors[0].lower()


def test_the_scheduler_uses_the_client_library_by_default() -> None:
    """Every cook's farm calls come from this factory, so without it no cook could connect."""
    scheduler: Any = REGISTRATION.DeadlineCloudPDGScheduler(None, "deadlinecloudscheduler1")
    assert isinstance(scheduler._clients(), REGISTRATION._ClientFactory)


class FakeCore:
    """The cook's ``on_schedule``: records each handle, then answers ``result`` or raises ``error``."""

    def __init__(self, result: Any, error: Optional[Exception]) -> None:
        self.result: Any = result
        self.error: Optional[Exception] = error
        self.handles: List[Any] = []

    def on_schedule(self, handle: Any) -> Any:
        self.handles.append(handle)
        if self.error is not None:
            raise self.error
        return self.result

    def on_stop_cook(self, cancel: bool) -> bool:
        return True


_ITEM: Any = types.SimpleNamespace(
    id=1, index=0, name="sim_1", node=types.SimpleNamespace(name="sim"), expectedOutputFiles=[]
)


@hypothesis_settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    result=st.sampled_from(list(ScheduleResult) + [None]),
    error=st.sampled_from(
        [None, RuntimeError("boom"), KeyError("x"), CookSetupError("setup"), ValueError("v")]
    ),
)
def test_schedule_answers_succeeded_or_deferred_from_the_cook_and_fails_anything_else(
    scene: FakeHou, result: Any, error: Optional[Exception]
) -> None:
    """PDG must hear Succeeded or Deferred exactly when the cook says so, and never an exception."""
    scheduler: Any = _scheduler(build_network(PYRO_BONFIRE))
    scheduler._core = FakeCore(result, error)
    answer: Any = scheduler.onSchedule(_ITEM)
    expected: Dict[Any, str] = {
        ScheduleResult.SUCCEEDED: PDG.scheduleResult.Succeeded,
        ScheduleResult.DEFERRED: PDG.scheduleResult.Deferred,
    }
    wanted: str = (
        expected.get(result, PDG.scheduleResult.CookFailed)
        if error is None
        else PDG.scheduleResult.CookFailed
    )
    assert answer == wanted
    # A raised error is shown on the node, and nothing else is.
    shown: bool = any(call[0] == "cookError" for call in scheduler.calls)
    assert shown is (error is not None)


def test_a_work_item_reaches_the_cook_with_its_node_and_declared_outputs(scene: FakeHou) -> None:
    """The cook names a task by its node and checks its declared outputs, so neither may be lost."""
    # Schedule one item with two outputs and one output with no path, through a recording core.
    scheduler: Any = _scheduler(build_network(PYRO_BONFIRE))
    core: FakeCore = FakeCore(ScheduleResult.SUCCEEDED, None)
    scheduler._core = core
    outputs: List[Any] = [
        types.SimpleNamespace(path="/out/a.bgeo", tag="file/geo"),
        types.SimpleNamespace(path="", tag="file/geo"),
        types.SimpleNamespace(path="/out/b.exr"),
    ]
    item: Any = types.SimpleNamespace(
        id=7,
        index=2,
        name="sim_7",
        node=types.SimpleNamespace(name="sim"),
        expectedOutputFiles=outputs,
    )
    scheduler.onSchedule(item)
    ((handle,),) = [core.handles]
    assert (handle.item_id, handle.index, handle.name, handle.node_name) == (7, 2, "sim_7", "sim")
    assert [(f.path, f.tag) for f in handle.declared_outputs] == [
        ("/out/a.bgeo", "file/geo"),
        ("/out/b.exr", ""),
    ]


def test_a_cook_starts_connected_with_the_cook_set_as_names(
    scene: FakeHou, tmp_path: Path, caplog: Any
) -> None:
    """PDG gives its cook set as node objects, and the cook needs their names to know what to wait for."""
    # Start with one PDG node in the cook set, then tick and stop the open cook.
    network: Network = build_network(PYRO_BONFIRE)
    scheduler: Any = _scheduler(network)
    with caplog.at_level(logging.INFO):
        assert scheduler.onStartCook(False, [network.node("sim")]) is True
    assert not [call for call in scheduler.calls if call[0] == "cookError"]
    assert "PDG asked to cook 1 node(s)." in caplog.text
    assert scheduler.onTick() == PDG.tickResult.SchedulerReady
    assert scheduler.onStopCook(False) is True


def test_an_empty_cook_set_is_not_read_as_cooking_nothing(scene: FakeHou, caplog: Any) -> None:
    """An empty set would leave every node out, so the cook treats it as unknown."""
    scheduler: Any = _scheduler(build_network(PYRO_BONFIRE))
    with caplog.at_level(logging.INFO):
        assert scheduler.onStartCook(False, []) is True
    assert "PDG asked to cook" not in caplog.text
    scheduler.onStopCook(True)


def test_a_failure_inside_a_callback_is_shown_on_the_node_and_never_raised(
    scene: FakeHou,
) -> None:
    """An exception reaching PDG would abort Houdini's cook without telling the artist why."""
    network: Network = build_network(PYRO_BONFIRE)
    scheduler: Any = _scheduler(network)
    scheduler.onStartCook(False, None)
    # A work item with no id cannot become a handle.
    assert scheduler.onSchedule(object()) == PDG.scheduleResult.CookFailed
    assert [call for call in scheduler.calls if call[0] == "cookError"]
    scheduler.onStopCook(True)


def test_the_scheduler_registers_under_the_hdas_type_name() -> None:
    """The HDA binds to the scheduler by this name, so a different one leaves the node without code."""
    registered: List[Tuple[Any, Dict[str, Any]]] = []

    class Registry:
        def registerScheduler(self, cls: Any, **kwargs: Any) -> None:
            registered.append((cls, kwargs))

    REGISTRATION.registerTypes(Registry())
    ((cls, kwargs),) = registered
    assert cls is REGISTRATION.DeadlineCloudPDGScheduler
    assert kwargs["name"] == "deadlinecloudscheduler" == cls.templateName()


# -- the composition root -----------------------------------------------------


def test_the_cook_uses_the_queues_bucket_and_checks_the_role_can_list_it(
    scene: FakeHou, tmp_path: Path
) -> None:
    """Payloads go to the queue's own bucket, and one read of a missing key proves the role can list it."""
    cloud: FakeCloud = FakeCloud()
    _connect(build_network(PYRO_BONFIRE), cloud, tmp_path)
    ((bucket, key),) = cloud.s3.keys
    assert bucket == "bucket-a" and key.startswith("DeadlineCloud/")
    # The queue's display name reaches the session, which names the queue in its errors.
    assert cloud.session_names == ["Render"]


def test_a_queue_role_that_cannot_list_its_bucket_is_refused_before_any_upload(
    scene: FakeHou, tmp_path: Path
) -> None:
    """Without the listing permission a missing output record reads as an error, so the cook can't tell results apart."""
    cloud: FakeCloud = FakeCloud(s3_error=_client_error("AccessDenied"))
    with pytest.raises(CookSetupError) as raised:
        _connect(build_network(PYRO_BONFIRE), cloud, tmp_path)
    assert "s3:ListBucket" in str(raised.value)
    assert "bucket-a" in str(raised.value) and "Render" in str(raised.value)


def test_another_error_from_the_bucket_check_is_logged_and_the_cook_goes_on(
    scene: FakeHou, tmp_path: Path, caplog: Any
) -> None:
    """A throttle or a network blip says nothing about the role, so it must not block the cook."""
    cloud: FakeCloud = FakeCloud(s3_error=_client_error("SlowDown"))
    with caplog.at_level(logging.WARNING):
        _connect(build_network(PYRO_BONFIRE), cloud, tmp_path)
    assert "Could not check access to bucket bucket-a" in caplog.text


def test_the_cooks_outcome_event_goes_through_the_telemetry_client(
    scene: FakeHou, tmp_path: Path
) -> None:
    """The client library's telemetry client honors the opt-out, so the event must go through it."""
    cloud: FakeCloud = FakeCloud()
    services: CookServices = _connect(build_network(PYRO_BONFIRE), cloud, tmp_path)
    services.journal.closed(CookSummary(cook_id="cook-1", final_phase=CookPhase.DONE))
    assert len(cloud.events) == 1


def _restore_telemetry_client(previous: Any) -> None:
    """Put back the library's cached telemetry client, or drop the one a test built."""
    _TELEMETRY_CLIENTS.pop("deadline-cloud-library", None)
    if previous is not None:
        _TELEMETRY_CLIENTS["deadline-cloud-library"] = previous


def test_an_opted_out_workstation_sends_no_outcome_event(
    scene: FakeHou, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: Any
) -> None:
    """The opt-out lives in the client library's own client, so the cook's event must go through it."""
    from queue import Queue

    from deadline.client import api

    # A fresh library client, built with the opt-out set, as on the artist's workstation.
    monkeypatch.setenv("DEADLINE_CLOUD_TELEMETRY_OPT_OUT", "true")
    previous: Any = _TELEMETRY_CLIENTS.pop("deadline-cloud-library", None)
    request.addfinalizer(lambda: _restore_telemetry_client(previous))
    client: Any = api.get_deadline_cloud_library_telemetry_client()
    client.event_queue = Queue()

    class Clients(FakeCloud):
        record_event: Any = REGISTRATION._ClientFactory.record_event

    record: Any = Clients().record_event()
    assert record.__self__ is client and record.__func__ is type(client).record_event
    services: CookServices = _connect(build_network(PYRO_BONFIRE), Clients(), tmp_path)
    services.journal.closed(CookSummary(cook_id="cook-1", final_phase=CookPhase.DONE))
    assert client.event_queue.empty()
    # The same client with telemetry on queues the event, so an empty queue means the opt-out held.
    monkeypatch.setattr(client, "telemetry_opted_out", False)
    monkeypatch.setattr(client, "_initialized", True)
    opted_in: CookServices = _connect(build_network(PYRO_BONFIRE), Clients(), tmp_path)
    opted_in.journal.closed(CookSummary(cook_id="cook-1", final_phase=CookPhase.DONE))
    assert not client.event_queue.empty()


@pytest.mark.parametrize(
    "settings, words",
    [({}, "farm ID"), ({"defaults.farm_id": "farm-1"}, "queue ID")],
)
def test_a_missing_farm_or_queue_is_named(
    scene: FakeHou, tmp_path: Path, settings: Dict[str, str], words: str
) -> None:
    """The artist needs to know which workstation setting to fill in."""
    with pytest.raises(CookSetupError, match=words):
        _connect(build_network(PYRO_BONFIRE), FakeCloud(settings=settings), tmp_path)


def test_an_expired_login_is_retried_while_it_renews(scene: FakeHou, tmp_path: Path) -> None:
    """The monitor renews credentials in the background, so a short wait often lets the read succeed."""
    cloud: FakeCloud = FakeCloud(
        queue=[NoCredentialsError(), _client_error("ExpiredToken"), _QUEUE]
    )
    _connect(build_network(PYRO_BONFIRE), cloud, tmp_path)
    assert cloud.sleeps == [2.0, 4.0]


@pytest.mark.parametrize(
    "error, words",
    [
        (NoCredentialsError(), "login is missing or has expired"),
        (_client_error("AccessDeniedException"), "was denied"),
        (_client_error("ResourceNotFoundException"), "was not found"),
        (RuntimeError("boom"), "Cannot read queue queue-1"),
    ],
)
def test_a_failed_queue_read_names_its_cause(
    scene: FakeHou, tmp_path: Path, error: Exception, words: str
) -> None:
    """Each cause has its own remedy, so the message must say which one it is."""
    with pytest.raises(CookSetupError, match=words):
        _connect(build_network(PYRO_BONFIRE), FakeCloud(queue=[error]), tmp_path)


def test_an_error_the_library_says_needs_a_login_asks_for_one(
    scene: FakeHou, tmp_path: Path
) -> None:
    """Some credential providers raise their own errors, so the library's login check decides."""
    cloud: FakeCloud = FakeCloud(queue=[RuntimeError("provider failed")], login_needed=True)
    with pytest.raises(CookSetupError, match="login is missing or has expired"):
        _connect(build_network(PYRO_BONFIRE), cloud, tmp_path)


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: Any) -> None:
    """The real client library with its own config file and a made-up key, so nothing reads this machine."""
    from deadline.client.api._session import invalidate_boto3_session_cache

    # Every file the library or boto3 reads is under tmp_path, and the key works only offline.
    monkeypatch.setenv("DEADLINE_CONFIG_FILE_PATH", str(tmp_path / "deadline" / "config"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws" / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws" / "credentials"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDWORKSTATION")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "workstation-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    for name in ("AWS_PROFILE", "AWS_SESSION_TOKEN", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    # The library caches its sessions, so each test starts and ends with none.
    invalidate_boto3_session_cache()
    request.addfinalizer(invalidate_boto3_session_cache)


def test_the_factory_reads_a_workstation_setting_and_an_unset_one_as_empty(library: None) -> None:
    """An unset farm ID must read as empty, or the cook can't tell the artist to set it."""
    from deadline.client import config

    config.set_setting("defaults.farm_id", "farm-1")
    clients: Any = REGISTRATION._ClientFactory()
    assert clients.get_setting("defaults.farm_id") == "farm-1"
    assert clients.get_setting("defaults.queue_id") == ""
    # A library that answers None for an unset value reads as empty too.
    clients.config = types.SimpleNamespace(get_setting=lambda name: None)
    assert clients.get_setting("defaults.farm_id") == ""


def test_the_factory_builds_a_deadline_client_and_a_queue_session_from_the_library(
    library: None,
) -> None:
    """Without these the cook can't read the queue, and its uploads would use the artist's own role."""
    from botocore.stub import Stubber

    clients: Any = REGISTRATION._ClientFactory()
    client: Any = clients.deadline_client()
    assert client.meta.service_model.service_name == "deadline"
    # The session's credentials come from the queue role that this client assumes for this queue.
    expires: datetime = datetime(2099, 1, 1, tzinfo=timezone.utc)
    stub: Stubber = Stubber(client)
    stub.add_response(
        "assume_queue_role_for_user",
        {
            "credentials": {
                "accessKeyId": "AKIDQUEUE",
                "secretAccessKey": "queue-secret",
                "sessionToken": "queue-token",
                "expiration": expires,
            }
        },
        {"farmId": "farm-1", "queueId": "queue-1"},
    )
    with stub:
        session: Any = clients.queue_session(client, "farm-1", "queue-1", "Render")
        assert session.get_credentials().get_frozen_credentials().access_key == "AKIDQUEUE"
    stub.assert_no_pending_responses()


class _AuthCheck:
    """The library's ``api`` as the login check sees it: the real status enum and a fixed answer."""

    def __init__(self, answer: Any) -> None:
        from deadline.client import api

        self.AwsAuthenticationStatus: Any = api.AwsAuthenticationStatus
        self.answer: Any = answer

    def check_authentication_status(self) -> Any:
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _statuses() -> List[Any]:
    """Every status the client library's login check can give."""
    from deadline.client import api

    return list(api.AwsAuthenticationStatus)


@pytest.mark.parametrize("status", _statuses(), ids=lambda status: status.name)
def test_only_a_status_that_needs_a_login_asks_the_artist_to_log_in(status: Any) -> None:
    """Asking for a login on any other status sends an artist with a working login the wrong way."""
    clients: Any = REGISTRATION._ClientFactory()
    clients.api = _AuthCheck(status)
    assert clients.needs_login() is (status.name == "NEEDS_LOGIN")


def test_a_login_check_that_fails_does_not_ask_for_a_login() -> None:
    """A failed check shows nothing about the login, so the queue error is reported as it is."""
    clients: Any = REGISTRATION._ClientFactory()
    clients.api = _AuthCheck(RuntimeError("no network"))
    assert clients.needs_login() is False


@pytest.mark.parametrize(
    "attachments", [{}, {"s3BucketName": "bucket-a"}, {"rootPrefix": "DeadlineCloud"}]
)
def test_a_queue_without_job_attachments_cannot_cook(
    scene: FakeHou, tmp_path: Path, attachments: Dict[str, str]
) -> None:
    """Every payload is stored under the bucket and its prefix, so a queue missing either is refused by name."""
    queue: Dict[str, Any] = {"displayName": "Render", "jobAttachmentSettings": attachments}
    cloud: FakeCloud = FakeCloud(queue=[queue])
    with pytest.raises(CookSetupError, match="Render has no job attachments bucket"):
        _connect(build_network(PYRO_BONFIRE), cloud, tmp_path)
