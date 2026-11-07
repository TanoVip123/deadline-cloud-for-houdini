# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The registration module, which builds a cook's objects, against fakes of ``pdg``, ``hou`` and Deadline Cloud."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from .houdini_fakes import (
    COOKING_EXAMPLE,
    PDG,
    PYRO_BONFIRE,
    FakeHou,
    Network,
    build_network,
    install_hou,
)

from deadline.houdini_pdg.cook import CookServices, SystemClock
from deadline.houdini_pdg.errors import CookSetupError
from deadline.houdini_pdg.houdini.host import HoudiniPdgHost
from deadline.houdini_pdg.houdini.settings import SchedulerSettings
from deadline.houdini_pdg.ports import CookPhase, CookSummary

#: The registration module's source. PDG loads it by path, so the test does too.
_REGISTRATION: str = os.path.join(
    os.path.dirname(__file__),
    *("..", "..", "..", "src", "deadline", "houdini_pdg", "houdini_plugin"),
    *("pdg", "types", "deadlinecloud", "__init__.py"),
)


def _load() -> ModuleType:
    """The registration module, loaded from its file the way PDG loads it."""
    spec: Any = importlib.util.spec_from_file_location("deadlinecloud_under_test", _REGISTRATION)
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REGISTRATION: ModuleType = _load()


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


class FakeCloud(REGISTRATION.DeadlineCloud):  # type: ignore[misc, name-defined]
    """Deadline Cloud with a fixed farm, queue, S3 answer, and telemetry, and no waiting."""

    def __init__(
        self,
        *,
        settings: Optional[Dict[str, str]] = None,
        queue: Optional[List[Any]] = None,
        s3_error: Optional[Exception] = _client_error("NoSuchKey"),
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

    def get_setting(self, name: str) -> str:
        return self.settings.get(name, "")

    def deadline_client(self) -> Any:
        return self.client

    def queue_session(self, client: Any, farm_id: str, queue_id: str, display_name: str) -> Any:
        return type("Session", (), {"client": lambda _self, name: self.s3})()

    def needs_login(self) -> bool:
        return False

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
        def _cloud(self) -> Any:
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
        settings, "cook-1", host=host, clock=SystemClock(), cloud=cloud
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


def test_a_cook_starts_connected_with_the_cook_set_as_names(
    scene: FakeHou, tmp_path: Path, caplog: Any
) -> None:
    """PDG gives its cook set as node objects, and the cook needs their names to know what to wait for."""
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


def test_a_queue_without_job_attachments_cannot_cook(scene: FakeHou, tmp_path: Path) -> None:
    """Every payload is stored in the bucket, so a queue without one is refused by name."""
    cloud: FakeCloud = FakeCloud(queue=[{"displayName": "Render"}])
    with pytest.raises(CookSetupError, match="Render has no job attachments bucket"):
        _connect(build_network(PYRO_BONFIRE), cloud, tmp_path)
