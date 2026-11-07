# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The scheduler node's Submit button: check the scene, confirm once, and cook the network.

The job's settings are the node's own parms, so Submit opens no settings dialog
and writes no parm, except the auto-save toggle when the artist picks Always Save.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, List, Optional, Tuple

import hou  # type: ignore[import-not-found]

from ..errors import CookSetupError
from ..validation import Verdict
from .host import PreCookChecks, prompt_to_save, read_scheduler_settings
from .settings import SchedulerSettings

_logger: logging.Logger = logging.getLogger(__name__)

_TITLE: str = "Deadline Cloud"
_CONFIRM_BUTTONS: Tuple[str, ...] = ("Submit", "Cancel")


def submit(node: Any, *, get_setting: Optional[Callable[[str], str]] = None) -> bool:
    """Run the node's Submit. Return True when the cook started.

    First the unsaved-scene check, then the network's default scheduler and the
    pre-cook checks, then one confirmation, then a cook of the display node.
    """
    from deadline.client.exceptions import UserInitiatedCancel

    read: Callable[[str], str] = get_setting or _workstation_setting
    # The save prompt comes first, so the artist answers it before anything else.
    try:
        prompt_to_save(node)
    except UserInitiatedCancel:
        return False
    network: Any = node.parent()
    problem: str = _default_scheduler_problem(node, network) or _farm_problem(read)
    if problem:
        _show(problem, error=True)
        return False
    verdict: Verdict = PreCookChecks().run(node, interactive=False)
    if verdict.refusals:
        _show(
            "The cook can't start:\n\n"
            + "\n\n".join(f"{r.subject}: {r.detail} {r.remedy}" for r in verdict.refusals),
            error=True,
        )
        return False
    try:
        settings: SchedulerSettings = read_scheduler_settings(node)
    except CookSetupError as e:
        _show(str(e), error=True)
        return False
    if not _confirmed(node, settings, verdict, read):
        return False
    # The display node is what the network's own Cook button cooks.
    network.displayNode().cookWorkItems(block=False)
    return True


def _default_scheduler_problem(node: Any, network: Any) -> str:
    """Why the network would not cook on ``node``, or ``""`` when its default scheduler is ``node``."""
    parm: Any = network.parm("topscheduler") if network is not None else None
    target: Any = network.node(parm.eval()) if parm is not None and parm.eval() else None
    if target is not None and target.path() == node.path():
        return ""
    return (
        f"The TOP network's Default TOP Scheduler is not {node.path()}, so the network would "
        "cook on another scheduler. Set the network's Default TOP Scheduler to this node."
    )


def _farm_problem(read: Callable[[str], str]) -> str:
    """The ROP's message for a missing farm or queue, or ``""``."""
    from ..submitter import _NO_FARM, _NO_QUEUE

    if not read("defaults.farm_id"):
        return _NO_FARM
    if not read("defaults.queue_id"):
        return _NO_QUEUE
    return ""


def _confirmed(
    node: Any, settings: SchedulerSettings, verdict: Verdict, read: Callable[[str], str]
) -> bool:
    """Show what will be submitted, with any advisory, and return whether the artist chose Submit."""
    lines: List[str] = [
        f"Farm: {read('defaults.farm_id')}",
        f"Queue: {read('defaults.queue_id')}",
        f"Job name: {_text_parm(node, 'name') or os.path.basename(str(hou.hipFile.path()))}",
        f"Priority: {_text_parm(node, 'priority') or '50'}",
        f"Working directory: {settings.working_dir}",
    ]
    details: str = "\n\n".join(f"{a.subject}: {a.detail}" for a in verdict.advisories)
    choice: int = hou.ui.displayMessage(
        "Submit this TOP network to Deadline Cloud?\n\n" + "\n".join(lines),
        title=_TITLE,
        buttons=_CONFIRM_BUTTONS,
        default_choice=0,
        close_choice=1,
        details=details or None,
    )
    return choice == 0


def _text_parm(node: Any, name: str) -> str:
    """A parm's value as text, or ``""`` when the node lacks it."""
    parm: Any = node.parm(name)
    return str(parm.evalAsString()) if parm is not None else ""


def _show(text: str, *, error: bool) -> None:
    """Tell the artist why nothing was submitted."""
    hou.ui.displayMessage(
        text, title=_TITLE, severity=hou.severityType.Error if error else hou.severityType.Message
    )


def _workstation_setting(name: str) -> str:
    """A Deadline Cloud workstation setting, such as ``defaults.farm_id``."""
    from deadline.client.config import get_setting

    return str(get_setting(name) or "")
