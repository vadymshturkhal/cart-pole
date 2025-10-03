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

        # Form area
        self.form = QFormLayout()
        layout.addLayout(self.form)

        # Buttons
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)

        # build form for initial agent
        self._rebuild_form(current_agent)

    # ---- internals ----
    def _on_agent_changed(self, name: str):
        # cache current edits
        self._cache[self.agent_name] = self._collect_values()
        self.agent_name = name
        if name not in self._cache:
            AgentClass = AGENTS[name]
            self._cache[name] = AgentClass.get_default_params()
        self._rebuild_form(name)

    def _clear_form(self):
        while self.form.count():
            item = self.form.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.params_widgets.clear()

    def _rebuild_form(self, agent_name: str):
        self._clear_form()
        spec = AGENT_SPECS[agent_name]
        values = self._cache[agent_name]
        for key, desc in spec.items():
            kind = desc[0]
            if kind == "bool":
                w = QCheckBox(); w.setChecked(bool(values.get(key, False)))
            elif kind == "int":
                _, lo, hi, step, default = desc
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
                w.setSingleStep(int(step))
                w.setValue(int(values.get(key, default)))
            elif kind == "float":
                _, lo, hi, step, default = desc
                w = QDoubleSpinBox()
                w.setRange(float(lo), float(hi))
                w.setSingleStep(float(step))
                w.setDecimals(6)
                w.setValue(float(values.get(key, default)))
            else:
                continue
            self.params_widgets[key] = w
            self.form.addRow(f"{key}:", w)

    def _collect_values(self) -> dict:
        out = {}
        for k, w in self.params_widgets.items():
            if isinstance(w, QCheckBox):
                out[k] = bool(w.isChecked())
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                out[k] = w.value()
        return out

    # ---- public API ----
    def get_selection(self) -> tuple[str, dict]:
        self._cache[self.agent_name] = self._collect_values()
        return self.agent_name, self._cache[self.agent_name].copy()
