from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QListWidget, QWidget,
    QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton,
)
from utils.agent_specs import AGENT_SPECS
from utils.agent_factory import AGENTS
import config


class AgentDialog(QDialog):
    def __init__(self, parent=None, current_agent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Agent")
        self.resize(420, 520)

        # cCache for per-agent hyperparams
        self._cache: dict[str, dict] = {}
        self.params_widgets: dict[str, QWidget] = {}

        # Pick initial agent
        if current_agent is None:
            current_agent = config.DEFAULT_AGENT

        self.agent_name = current_agent

        # Load default params from Agent class
        AgentClass = AGENTS[self.agent_name]
        self._cache[self.agent_name] = AgentClass.get_default_hyperparams()

        layout = QVBoxLayout(self)

        # Agent list
        self.agent_list = QListWidget()
        self.agent_list.addItems(list(AGENT_SPECS.keys()))
        self.agent_list.setCurrentRow(list(AGENT_SPECS.keys()).index(current_agent))
        self.agent_list.currentTextChanged.connect(self._on_agent_changed)
        layout.addWidget(self.agent_list)

        # Buttons
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)

    def _on_agent_changed(self, name: str):
        self._cache[self.agent_name] = self._collect_values()
        self.agent_name = name
        if name not in self._cache:
            AgentClass = AGENTS[name]
            self._cache[name] = AgentClass.get_default_hyperparams()

    def _collect_values(self) -> dict:
        out = {}
        for k, w in self.params_widgets.items():
            if isinstance(w, QCheckBox):
                out[k] = bool(w.isChecked())
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                out[k] = w.value()
        return out

    def get_selection(self) -> str:
        self._cache[self.agent_name] = self._collect_values()
        return self.agent_name
