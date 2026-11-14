# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The scheduler node's Parameters panel: the farm, queue and storage profile above its parms.

Only the node's Python panel imports this module. Houdini rebuilds the panel on every selection,
so a build only makes widgets, and the lists load in deadline-cloud's own background threads.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import hou  # type: ignore[import-not-found]
from deadline.client.ui.deadline_authentication_status import DeadlineAuthenticationStatus
from deadline.client.ui.widgets import DeadlineCloudSettingsWidget
from qtpy.QtCore import QCoreApplication  # type: ignore
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget  # type: ignore

LOGIN_TEXT: str = (
    "Log in with Deadline Cloud Monitor or `deadline auth login`, "
    "then press the refresh button next to Farm."
)
CHECKING_TEXT: str = "Checking your Deadline Cloud login."
# The status line for each login state. None means deadline-cloud is still checking.
_LOGIN_STATUS_TEXT: Dict[Optional[bool], str] = {True: "", False: LOGIN_TEXT, None: CHECKING_TEXT}


def _pump_qt_events() -> None:
    """Deliver Qt's posted events, which carry the lists back from their threads."""
    QCoreApplication.sendPostedEvents(None, 0)


# Houdini's own event loop doesn't deliver Qt's cross-thread signals, so drain them on each tick.
# The module loads once per session, so the pump and the shared login status are made once.
hou.ui.addEventLoopCallback(_pump_qt_events)
_AUTH_STATUS: Any = DeadlineAuthenticationStatus.getInstance()


class SchedulerPanel(QWidget):
    """deadline-cloud's farm, queue and storage profile lists, a status line, and the node's parms."""

    def __init__(self, node: Any) -> None:
        super().__init__()
        self._node: Any = node
        self._status: Any = QLabel("")
        self._status.setWordWrap(True)
        layout: Any = QVBoxLayout(self)
        # A widget that fails costs only the dropdowns, never the parms.
        self._settings: Optional[Any] = None
        try:
            # The slots are this panel's methods, so Qt drops them when Houdini drops the panel.
            self._settings = DeadlineCloudSettingsWidget(parent=self)
            self._settings.selection_changed.connect(self._update_node)
            for box in (
                self._settings.farm_box,
                self._settings.queue_box,
                self._settings.storage_profile_box,
            ):
                box.background_exception.connect(self._show_list_error)
            _AUTH_STATUS.api_availability_changed.connect(self._refresh_lists)
            layout.addWidget(self._settings)
        except Exception as error:
            self._settings = None
            self._status.setText(f"The farm and queue lists couldn't be shown: {error}")
        layout.addWidget(self._status)
        self.editor: Any = hou.qt.ParmDialog(node, showTitleBar=False)
        layout.addWidget(self.editor)
        if self._settings is not None:
            self._refresh_lists()

    def _refresh_lists(self) -> None:
        """Show the saved picks, and load the lists only while the Deadline Cloud API answers."""
        available: Optional[bool] = _AUTH_STATUS.api_availability
        try:
            self._settings.refresh_setting_controls(available is True)  # type: ignore[union-attr]
            self._status.setText(_LOGIN_STATUS_TEXT[available])
        except Exception as error:
            self._status.setText(f"The farm and queue lists couldn't be refreshed: {error}")

    def _show_list_error(self, title: str, error: BaseException) -> None:
        """Show a list that failed to load, since deadline-cloud shows no popup for it."""
        self._status.setText(f"{title} failed: {error}")

    def _update_node(self) -> None:
        """Show the new farm and queue on the node, as its Update Queue Parameters button does."""
        # The update calls the API on the main thread, so an error here must not leave the slot.
        try:
            self._node.hdaModule().update_queue_parameters_callback({"node": self._node})
            self._status.setText("")
        except Exception as error:
            self._status.setText(f"The node's farm and queue couldn't be updated: {error}")


def create_panel(node: Any) -> Tuple[Any, Optional[Any]]:
    """Return the panel for ``node`` and its parameter editor, or an error label and None."""
    try:
        panel: SchedulerPanel = SchedulerPanel(node)
    except Exception as error:
        return QLabel(f"The node's parameters couldn't be shown: {error}"), None
    return panel, panel.editor
