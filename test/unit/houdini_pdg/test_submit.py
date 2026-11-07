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

from deadline.houdini_pdg.houdini.submit import submit

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


def test_a_refused_graph_is_named_and_not_cooked(scene: FakeHou) -> None:
    """The artist sees why before anything is uploaded, and no confirmation is offered."""
    network: Network = build_network(
        COOKING_EXAMPLE, scheduler_parms={"working_dir": "$HIP", "validate_graph": 1}
    )
    assert submit(network.scheduler_node, get_setting=_workstation) is False
    assert _cooks(network) == []
    ((text, buttons),) = scene.messages.calls
    assert "python1" in text and buttons == ()


def test_a_workstation_without_a_farm_is_told_which_setting_is_missing(scene: FakeHou) -> None:
    """With no farm the job has nowhere to go, so the ROP's message names the setting."""
    network: Network = _network()
    assert submit(network.scheduler_node, get_setting=lambda name: "") is False
    assert "farm ID" in scene.messages.calls[0][0]
    assert _cooks(network) == []
