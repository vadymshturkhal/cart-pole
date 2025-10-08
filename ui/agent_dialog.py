from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QListWidget, QFormLayout,
    QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QWidget
)
from utils.agent_specs import AGENT_SPECS   # still needed for UI spec (sliders, ranges)
from utils.agent_factory import AGENTS


class AgentDialog(QDialog):
    def __init__(self, parent=None, current_agent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Agent & Hyperparameters")
        self.resize(420, 520)

        # cache for per-agent hyperparams
        self._cache: dict[str, dict] = {}
        self.params_widgets: dict[str, QWidget] = {}

        # pick initial agent
        if current_agent is None:
            current_agent = list(AGENTS.keys())[0]
        self.agent_name = current_agent

        # load default params from Agent class
        AgentClass = AGENTS[self.agent_name]
        self._cache[self.agent_name] = AgentClass.get_default_hyperparams()

        # === UI layout ===
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
