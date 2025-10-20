from PySide6.QtWidgets import QWidget


class InlinePanelManager:
    """
    Manages inline panels that temporarily replace the reward/loss plots area.

    Used for Load Model, NN Config, Agent Config, Environment Config, etc.
    Handles layout replacement, cleanup, and restoring the original plots.
    """

    def __init__(self, ui):
        self.ui = ui
        self._active_panel: QWidget | None = None

    # --------------------------------------------------------------
    # Core panel management
    # --------------------------------------------------------------
    def show_panel(self, panel: QWidget):
        """
        Hide the plot tabs and insert the given panel at the top of the layout.
        If another panel is already active, it will be closed first.
        """
        if self._active_panel:
            self.close_panel()

        self.ui.tabs.setVisible(False)
        self.ui.layout.insertWidget(0, panel)
        self._active_panel = panel

    def close_panel(self):
        """
        Remove the currently active panel, delete it safely,
        and restore the plot tabs.
        """
        if not self._active_panel:
            return

        self.ui.layout.removeWidget(self._active_panel)
        self._active_panel.deleteLater()
        self._active_panel = None
        self.ui.tabs.setVisible(True)

    def has_active_panel(self) -> bool:
        """Return True if an inline panel is currently displayed."""
        return self._active_panel is not None
