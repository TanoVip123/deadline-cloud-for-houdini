# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The scheduler node's Parameters panel, against fakes of ``hou``, Qt and deadline-cloud's widget.

The test env has no Qt binding, so each test imports the panel module fresh over fake modules.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
import types
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from unittest.mock import patch

import pytest

# The panel module, and the plugin files the tests read: the pypanel and the HDA.
_PANEL_MODULE: str = "deadline.houdini_pdg.houdini.panel"
_PDG_SRC: str = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "src", "deadline", "houdini_pdg"
)
_PLUGIN: str = os.path.join(_PDG_SRC, "houdini_plugin")
_PYPANEL: str = os.path.join(_PLUGIN, "python_panels", "deadline_cloud_scheduler.pypanel")
_HDA: str = os.path.join(_PLUGIN, "otls", "deadline_cloud_scheduler.hda")


class FakeSignal:
    """A Qt signal: ``connect`` keeps the slot, and ``emit`` calls every slot."""

    def __init__(self) -> None:
        self.slots: List[Callable[..., Any]] = []

    def connect(self, slot: Callable[..., Any], *args: Any) -> None:
        self.slots.append(slot)

    def emit(self, *args: Any) -> None:
        for slot in list(self.slots):
            slot(*args)


class FakeWidget:
    """``QWidget``: keeps the widgets its layout holds, in order."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.children: List[Any] = []


class FakeLabel(FakeWidget):
    """``QLabel``: its text."""

    def __init__(self, text: str = "", *args: Any) -> None:
        super().__init__()
        self._text: str = text
        self.wraps: bool = False

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text

    def setWordWrap(self, wrap: bool) -> None:
        self.wraps = wrap


class FakeLayout:
    """``QVBoxLayout``: adds each widget to its parent's ``children``."""

    def __init__(self, parent: FakeWidget) -> None:
        self._parent: FakeWidget = parent

    def addWidget(self, widget: Any, *args: Any) -> None:
        self._parent.children.append(widget)


class FakeParmDialog(FakeWidget):
    """``hou.qt.ParmDialog``: the node it shows."""

    error: Optional[Exception] = None

    def __init__(self, node: Any, **kwargs: Any) -> None:
        super().__init__()
        if FakeParmDialog.error is not None:
            raise FakeParmDialog.error
        self.node: Any = node

    def setNode(self, node: Any) -> None:
        self.node = node


class FakeComboBox:
    """One of the widget's farm, queue and storage profile combos."""

    def __init__(self) -> None:
        self.background_exception: FakeSignal = FakeSignal()


class FakeSettingsWidget(FakeWidget):
    """deadline-cloud's ``DeadlineCloudSettingsWidget``, recording each refresh."""

    error: Optional[Exception] = None
    refresh_error: Optional[Exception] = None
    refreshes: List[bool] = []

    def __init__(self, *, parent: Any = None) -> None:
        super().__init__()
        # A test can make the constructor fail, as a broken deadline-cloud install would.
        if FakeSettingsWidget.error is not None:
            raise FakeSettingsWidget.error
        self.selection_changed: FakeSignal = FakeSignal()
        self.farm_box: FakeComboBox = FakeComboBox()
        self.queue_box: FakeComboBox = FakeComboBox()
        self.storage_profile_box: FakeComboBox = FakeComboBox()

    def refresh_setting_controls(self, deadline_authorized: bool) -> None:
        FakeSettingsWidget.refreshes.append(deadline_authorized)
        # A test can make a refresh fail, as a broken config file would.
        if FakeSettingsWidget.refresh_error is not None:
            raise FakeSettingsWidget.refresh_error


class FakeAuthStatus:
    """deadline-cloud's login status. ``refresh_status`` is its network probe."""

    def __init__(self) -> None:
        self.api_availability: Optional[bool] = None
        self.api_availability_changed: FakeSignal = FakeSignal()
        self.probes: int = 0

    def refresh_status(self) -> None:
        self.probes += 1


class FakeNode:
    """A scheduler node whose HDA module records the Update Queue Parameters callback."""

    def __init__(self, error: Optional[Exception] = None) -> None:
        self.updates: List[Dict[str, Any]] = []
        self.error: Optional[Exception] = error

    def hdaModule(self) -> Any:
        return types.SimpleNamespace(update_queue_parameters_callback=self._update)

    def _update(self, kwargs: Dict[str, Any]) -> None:
        self.updates.append(kwargs)
        if self.error is not None:
            raise self.error


class Fakes:
    """The fake modules, and what the panel did with them."""

    def __init__(self) -> None:
        # What the panel registers and posts, and a ``hou`` with only the two parts it uses.
        self.auth: FakeAuthStatus = FakeAuthStatus()
        self.loop_callbacks: List[Callable[[], None]] = []
        self.posted: List[Tuple[Any, int]] = []
        self.hou: types.ModuleType = types.ModuleType("hou")
        self.hou.qt = types.SimpleNamespace(ParmDialog=FakeParmDialog)  # type: ignore[attr-defined]
        self.hou.ui = types.SimpleNamespace(  # type: ignore[attr-defined]
            addEventLoopCallback=self.loop_callbacks.append
        )
        self.panel: Any = None

    def modules(self) -> Dict[str, Any]:
        """Every module the panel imports, as fakes."""
        # Qt's posted-event call records its arguments, and the widgets keep their children.
        core: types.ModuleType = types.ModuleType("qtpy.QtCore")
        core.QCoreApplication = types.SimpleNamespace(  # type: ignore[attr-defined]
            sendPostedEvents=lambda receiver, kind: self.posted.append((receiver, kind))
        )
        widgets: types.ModuleType = types.ModuleType("qtpy.QtWidgets")
        widgets.QWidget = FakeWidget  # type: ignore[attr-defined]
        widgets.QLabel = FakeLabel  # type: ignore[attr-defined]
        widgets.QVBoxLayout = FakeLayout  # type: ignore[attr-defined]
        # deadline-cloud's widget and login status.
        settings: types.ModuleType = types.ModuleType("deadline.client.ui.widgets")
        settings.DeadlineCloudSettingsWidget = FakeSettingsWidget  # type: ignore[attr-defined]
        auth: types.ModuleType = types.ModuleType(
            "deadline.client.ui.deadline_authentication_status"
        )
        auth.DeadlineAuthenticationStatus = types.SimpleNamespace(  # type: ignore[attr-defined]
            getInstance=lambda: self.auth
        )
        return {
            "hou": self.hou,
            "qtpy": types.ModuleType("qtpy"),
            "qtpy.QtCore": core,
            "qtpy.QtWidgets": widgets,
            "deadline.client.ui.widgets": settings,
            "deadline.client.ui.deadline_authentication_status": auth,
            # Any import of the login dialog fails, so a build that tried it would show an error.
            "deadline.client.ui.dialogs.deadline_login_dialog": None,
        }


@pytest.fixture(autouse=True)
def _reset_fake_classes() -> None:
    """Every test starts with widgets that build, and no recorded refreshes."""
    # The fakes keep these on the class, so one test's failure can't leak into the next.
    FakeSettingsWidget.error = None
    FakeSettingsWidget.refresh_error = None
    FakeSettingsWidget.refreshes = []
    FakeParmDialog.error = None


@pytest.fixture
def fakes() -> Iterator[Fakes]:
    """The panel module, imported fresh over the fakes, so its once-per-session state is new."""
    state: Fakes = Fakes()
    with patch.dict(sys.modules, state.modules()):
        sys.modules.pop(_PANEL_MODULE, None)
        state.panel = importlib.import_module(_PANEL_MODULE)
        yield state


def _read_status_line(widget: FakeWidget) -> str:
    """The text of the panel's status line."""
    labels: List[FakeLabel] = [child for child in widget.children if isinstance(child, FakeLabel)]
    assert len(labels) == 1
    return labels[0].text()


def test_the_panel_shows_the_widget_then_the_status_line_then_the_node_parms(fakes: Fakes) -> None:
    """The dropdowns sit above the node's parms, and the editor shows the selected node."""
    # The panel's children, top to bottom.
    node: FakeNode = FakeNode()
    widget: Any
    editor: Any
    widget, editor = fakes.panel.create_panel(node)
    assert [type(child) for child in widget.children] == [
        FakeSettingsWidget,
        FakeLabel,
        FakeParmDialog,
    ]
    assert widget.children[2] is editor and editor.node is node
    # A long error wraps instead of widening the pane.
    assert widget.children[1].wraps


def test_builds_make_no_network_call_and_register_the_pump_once(fakes: Fakes) -> None:
    """Houdini rebuilds the panel on every selection, so a build must not probe or add callbacks."""
    fakes.auth.api_availability = False
    for _ in range(3):
        fakes.panel.create_panel(FakeNode())
    assert fakes.auth.probes == 0
    assert FakeSettingsWidget.refreshes == [False, False, False]
    assert len(fakes.loop_callbacks) == 1
    # The pump delivers every posted Qt event, which carries the lists back from their threads.
    fakes.loop_callbacks[0]()
    assert fakes.posted == [(None, 0)]


def test_the_lists_load_once_the_api_is_available_and_again_when_it_changes(fakes: Fakes) -> None:
    """Lists load only for a working login, and a login made later fills them without a rebuild."""
    # Built logged out, the widget shows the saved picks and fetches nothing.
    fakes.auth.api_availability = False
    widget: Any
    widget, _ = fakes.panel.create_panel(FakeNode())
    assert FakeSettingsWidget.refreshes == [False]
    # A login reported later fetches the lists, and a logout after it stops fetching.
    fakes.auth.api_availability = True
    fakes.auth.api_availability_changed.emit()
    assert FakeSettingsWidget.refreshes == [False, True]
    assert _read_status_line(widget) == ""
    fakes.auth.api_availability = False
    fakes.auth.api_availability_changed.emit()
    assert FakeSettingsWidget.refreshes == [False, True, False]


def test_without_a_login_the_status_line_says_how_to_log_in(fakes: Fakes) -> None:
    """The panel has no Login button, so the status line names the two ways to log in."""
    fakes.auth.api_availability = False
    widget: Any
    widget, _ = fakes.panel.create_panel(FakeNode())
    assert "Deadline Cloud Monitor" in _read_status_line(widget)
    assert "deadline auth login" in _read_status_line(widget)
    assert "refresh" in _read_status_line(widget)
    # Logging in clears it.
    fakes.auth.api_availability = True
    fakes.auth.api_availability_changed.emit()
    assert _read_status_line(widget) == ""


def test_while_the_login_is_being_checked_the_status_line_says_so(fakes: Fakes) -> None:
    """Before deadline-cloud's first check ends, the panel doesn't claim the artist is logged out."""
    # The fake status starts as deadline-cloud's does, with no answer yet.
    widget: Any
    widget, _ = fakes.panel.create_panel(FakeNode())
    assert FakeSettingsWidget.refreshes == [False]
    assert "Checking" in _read_status_line(widget)
    assert "deadline auth login" not in _read_status_line(widget)


@pytest.mark.parametrize("box", ["farm_box", "queue_box", "storage_profile_box"])
def test_a_list_error_shows_in_the_status_line(fakes: Fakes, box: str) -> None:
    """deadline-cloud shows no popup for a failed list, so the status line is all the artist sees."""
    fakes.auth.api_availability = True
    widget: Any
    widget, _ = fakes.panel.create_panel(FakeNode())
    # The combo sends its title and the error, as deadline-cloud's do.
    settings: FakeSettingsWidget = widget.children[0]
    getattr(settings, box).background_exception.emit("Refresh Farms list", RuntimeError("denied"))
    status: str = _read_status_line(widget)
    assert "Refresh Farms list" in status and "denied" in status


def test_a_farm_or_queue_pick_updates_the_node_as_its_button_does(fakes: Fakes) -> None:
    """The Farm and Queue labels and queue parameters follow a pick, through the node's callback."""
    # The widget says a pick was saved, and the node runs its Update Queue Parameters code.
    node: FakeNode = FakeNode()
    widget: Any
    widget, _ = fakes.panel.create_panel(node)
    widget.children[0].selection_changed.emit()
    assert node.updates == [{"node": node}]


def test_a_failed_node_update_shows_in_the_status_line_and_raises_nothing(fakes: Fakes) -> None:
    """The update calls the API on the main thread, and its error must not leave a Qt slot."""
    fakes.auth.api_availability = True
    node: FakeNode = FakeNode(error=RuntimeError("expired token"))
    widget: Any
    widget, _ = fakes.panel.create_panel(node)
    widget.children[0].selection_changed.emit()
    assert "expired token" in _read_status_line(widget)
    # A later update that works clears the old error.
    node.error = None
    widget.children[0].selection_changed.emit()
    assert _read_status_line(widget) == ""


def test_a_refresh_that_fails_after_a_login_change_shows_in_the_status_line(fakes: Fakes) -> None:
    """The login signal calls the refresh from a Qt slot, so its error must not raise into Houdini."""
    # A good build, then a refresh that fails when the login changes.
    fakes.auth.api_availability = False
    widget: Any
    editor: Any
    widget, editor = fakes.panel.create_panel(FakeNode())
    FakeSettingsWidget.refresh_error = RuntimeError("bad config file")
    fakes.auth.api_availability = True
    fakes.auth.api_availability_changed.emit()
    assert "bad config file" in _read_status_line(widget)
    assert widget.children[-1] is editor


def test_a_widget_that_fails_to_build_still_leaves_the_node_parms(fakes: Fakes) -> None:
    """The panel replaces the parm pane, so losing the dropdowns must not lose Submit too."""
    # The widget's constructor fails, and the status line and the parms remain.
    FakeSettingsWidget.error = RuntimeError("no farms module")
    node: FakeNode = FakeNode()
    widget: Any
    editor: Any
    widget, editor = fakes.panel.create_panel(node)
    assert [type(child) for child in widget.children] == [FakeLabel, FakeParmDialog]
    assert editor.node is node
    assert "no farms module" in _read_status_line(widget)


def test_a_parm_editor_that_fails_shows_the_error_and_raises_nothing(fakes: Fakes) -> None:
    """Houdini's handling of a raise in a panel is unknown, so the last resort is a label."""
    # Without an editor, there's nothing for a new selection to go to.
    FakeParmDialog.error = RuntimeError("no editor")
    widget: Any
    editor: Any
    widget, editor = fakes.panel.create_panel(FakeNode())
    assert editor is None
    assert isinstance(widget, FakeLabel) and "no editor" in widget.text()


def _read_interface() -> ET.Element:
    """The pypanel file's one interface."""
    interfaces: List[ET.Element] = ET.parse(_PYPANEL).getroot().findall("interface")
    assert len(interfaces) == 1
    return interfaces[0]


def test_the_panel_file_names_the_scheduler_node_type_and_stays_out_of_the_menu() -> None:
    """A wrong optype shows the normal pane, and the ROP's name would replace the ROP's panel."""
    # The name is the panel's own, not the ROP's ``deadline_cloud``.
    interface: ET.Element = _read_interface()
    assert interface.get("name") == "deadline_cloud_scheduler"
    # The optype must match the HDA's table and operator, so renaming the HDA breaks this test.
    with open(os.path.join(_HDA, "INDEX__SECTION"), encoding="utf8") as f:
        index: Dict[str, str] = {
            key.strip(): value.strip()
            for key, value in (line.split(":", 1) for line in f if ":" in line)
        }
    shown: Optional[ET.Element] = interface.find("showInParametersPane")
    assert shown is not None
    assert shown.get("optype") == f"{index['Table']}/{index['Operator']}"
    assert interface.find("includeInToolbarMenu") is None


class FakePaneTab:
    """The Parameters pane tab Houdini hands ``onCreateInterface``."""

    def __init__(self, node: Any) -> None:
        self._node: Any = node

    def currentNode(self) -> Any:
        return self._node


class FakeTypedNode:
    """A node of one type, as Houdini hands it to ``onNodePathChanged``."""

    def __init__(self, type_name: str) -> None:
        self._type: Any = types.SimpleNamespace(name=lambda: type_name)

    def type(self) -> Any:
        return self._type


def _run_script(node: Any, modules: Dict[str, Any]) -> Dict[str, Any]:
    """Run the pypanel's script as Houdini does, over ``modules``, and return its names."""
    # Houdini runs the script in its own namespace, with ``kwargs`` naming the pane tab.
    script: Optional[str] = _read_interface().findtext("script")
    assert script is not None
    namespace: Dict[str, Any] = {"kwargs": {"paneTab": FakePaneTab(node)}}
    with patch.dict(sys.modules, modules):
        exec(compile(script, _PYPANEL, "exec"), namespace)
    return namespace


def _raise_no_login_status() -> Any:
    """``getInstance`` failing, as a broken Qt or file watcher would make it."""
    raise RuntimeError("no login status")


def test_the_script_builds_the_panel_and_follows_only_scheduler_nodes() -> None:
    """Later scheduler nodes go to the same editor, and other node types never reach it."""
    first: FakeTypedNode = FakeTypedNode("deadlinecloudscheduler")
    editor: FakeParmDialog = FakeParmDialog(first)
    # A stand-in panel module that records the node it was handed.
    built: List[Any] = []

    def create_panel(node: Any) -> Tuple[str, FakeParmDialog]:
        built.append(node)
        return "panel", editor

    panel: types.ModuleType = types.ModuleType(_PANEL_MODULE)
    panel.create_panel = create_panel  # type: ignore[attr-defined]
    fakes: Fakes = Fakes()
    script: Dict[str, Any] = _run_script(first, {"hou": fakes.hou, _PANEL_MODULE: panel})
    assert script["onCreateInterface"]() == "panel"
    assert built == [first]
    second: FakeTypedNode = FakeTypedNode("deadlinecloudscheduler")
    script["onNodePathChanged"](second)
    assert editor.node is second
    # A node of another type, or no node, leaves the editor on the last scheduler node.
    script["onNodePathChanged"](FakeTypedNode("box"))
    script["onNodePathChanged"](None)
    assert editor.node is second


def test_a_new_selection_after_the_parm_editor_failed_raises_nothing() -> None:
    """With no editor, Houdini still calls ``onNodePathChanged`` on each click, which must not raise."""

    # A stand-in panel module whose editor failed, so it returns only the error label.
    def create_panel(node: Any) -> Tuple[str, None]:
        return "label", None

    panel: types.ModuleType = types.ModuleType(_PANEL_MODULE)
    panel.create_panel = create_panel  # type: ignore[attr-defined]
    fakes: Fakes = Fakes()
    node: FakeTypedNode = FakeTypedNode("deadlinecloudscheduler")
    script: Dict[str, Any] = _run_script(node, {"hou": fakes.hou, _PANEL_MODULE: panel})
    assert script["onCreateInterface"]() == "label"
    script["onNodePathChanged"](FakeTypedNode("deadlinecloudscheduler"))


def test_a_panel_module_that_fails_to_import_leaves_the_node_parms_and_one_error_line() -> None:
    """A broken install must not cost the artist the node's parms and its Submit button."""
    # The real panel module fails as it loads, because the login status can't be made.
    fakes: Fakes = Fakes()
    modules: Dict[str, Any] = fakes.modules()
    modules["deadline.client.ui.deadline_authentication_status"].DeadlineAuthenticationStatus = (
        types.SimpleNamespace(getInstance=_raise_no_login_status)
    )
    # Houdini's own Qt module is there for the fallback.
    qt: types.ModuleType = types.ModuleType("hutil.Qt")
    qt.QtWidgets = types.SimpleNamespace(  # type: ignore[attr-defined]
        QWidget=FakeWidget, QLabel=FakeLabel, QVBoxLayout=FakeLayout
    )
    modules.update({"hutil": types.ModuleType("hutil"), "hutil.Qt": qt})
    # The panel module isn't cached, so the script's import runs its load code.
    with patch.dict(sys.modules, modules):
        sys.modules.pop(_PANEL_MODULE, None)
        script: Dict[str, Any] = _run_script("node", modules)
        widget: Any = script["onCreateInterface"]()
    assert [type(child) for child in widget.children] == [FakeLabel, FakeParmDialog]
    assert "no login status" in widget.children[0].text()
    assert widget.children[1].node == "node"
    other: FakeTypedNode = FakeTypedNode("deadlinecloudscheduler")
    script["onNodePathChanged"](other)
    assert widget.children[1].node is other


def _list_imports(path: str) -> List[str]:
    """Every module a Python file imports, with relative imports resolved."""
    with open(path, encoding="utf8") as f:
        tree: ast.Module = ast.parse(f.read(), path)
    # The file's own package, which a relative import starts from.
    folder: str = os.path.relpath(os.path.dirname(path), os.path.dirname(_PDG_SRC))
    package: str = "deadline." + folder.replace(os.sep, ".")
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base: str = package.rsplit(".", node.level - 1)[0] if node.level else ""
            module: str = ".".join(part for part in (base, node.module or "") if part)
            names.append(module)
            names.extend(f"{module}.{alias.name}" for alias in node.names)
    return names


def test_no_module_hython_loads_imports_the_panel() -> None:
    """hython has no Qt UI, so the registration, cook and node scripts must never load the panel."""
    # Every Python file of the scheduler, and the node's own Python module.
    paths: List[str] = [
        os.path.join(folder, name)
        for folder, _, files in os.walk(_PDG_SRC)
        for name in files
        if name.endswith(".py") and name != "panel.py"
    ]
    paths.append(os.path.join(_HDA, "Top_1deadlinecloudscheduler", "PythonModule"))
    # The scan must reach the registration module, or it proves nothing about the cook.
    assert any(path.endswith(os.path.join("deadlinecloud", "__init__.py")) for path in paths)
    importers: List[str] = [path for path in paths if _PANEL_MODULE in _list_imports(path)]
    assert importers == []
