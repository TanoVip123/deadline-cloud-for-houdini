# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The scheduler node's Submit button, against small fakes of ``hou`` and ``pdg``."""

from __future__ import annotations

import os
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
