from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QPushButton, QLabel, QHBoxLayout
)
from utils.agent_specs import AGENT_SPECS
import config


class AgentSelectPanel(QWidget):
    """Inline panel for selecting an agent."""

    def __init__(self, on_close_callback, current_agent=None, read_only=False):
        super().__init__()
        self.on_close_callback = on_close_callback
        self.read_only = read_only

        if current_agent is None:
            current_agent = config.DEFAULT_AGENT
        self.agent_name = current_agent
        self._cache: dict[str, dict] = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("<b>🤖 Choose Agent</b>")
        title.setStyleSheet("font-size:16px; color:#ddd;")
        layout.addWidget(title)

        self.agent_list = QListWidget()
        self.agent_list.addItems(list(AGENT_SPECS.keys()))
        if current_agent in AGENT_SPECS:
            self.agent_list.setCurrentRow(list(AGENT_SPECS.keys()).index(current_agent))
        layout.addWidget(self.agent_list)

        # Buttons
        btn_row = QHBoxLayout()
        self.ok_btn = QPushButton("Select")
        self.cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(self.ok_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.ok_btn.clicked.connect(self._on_select)
        self.cancel_btn.clicked.connect(self._on_cancel)

        # Disable selection in read-only mode
        if self.read_only:
            self.agent_list.setEnabled(False)
            self.ok_btn.setEnabled(False)
            layout.addWidget(QLabel("<span style='color:#bbb;'>🔒 Read-only mode (Training in progress)</span>"))

    def _on_select(self):
        self.agent_name = self.agent_list.currentItem().text()
        self.on_close_callback(True, self.agent_name)

    def _on_cancel(self):
        self.on_close_callback(False, None)
