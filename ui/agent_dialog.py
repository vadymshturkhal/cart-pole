# ui/agent_dialog.py
from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QFormLayout, \
    QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QWidget
from utils.agent_specs import AGENT_SPECS, default_hparams

class AgentDialog(QDialog):
    def __init__(self, parent=None, current_agent="nstep_dqn", defaults: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Select Agent & Hyperparameters")
        self.resize(420, 520)

        self._cache: dict[str, dict] = {}   # remember per-agent edits
        self.agent_name = current_agent
        self.params_widgets: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)

        # Agent list
        self.agent_list = QListWidget()
        self.agent_list.addItems(list(AGENT_SPECS.keys()))
        # preselect
        names = list(AGENT_SPECS.keys())
        self.agent_list.setCurrentRow(max(0, names.index(current_agent)))
        self.agent_list.currentTextChanged.connect(self._on_agent_changed)
        layout.addWidget(self.agent_list)

        # Form area
        self.form = QFormLayout()
        layout.addLayout(self.form)

        # Buttons
        ok_btn = QPushButton("OK"); ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)

        # seed form
        if defaults is None:
            defaults = default_hparams(current_agent)
        self._cache[current_agent] = defaults.copy()
        self._rebuild_form(current_agent)

    # ---- internals ----
    def _on_agent_changed(self, name: str):
        # cache current edits
        self._cache[self.agent_name] = self._collect_values()
        self.agent_name = name
        if name not in self._cache:
            self._cache[name] = default_hparams(name)
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
                w = QSpinBox(); w.setRange(int(lo), int(hi)); w.setSingleStep(int(step)); w.setValue(int(values.get(key, default)))
            elif kind == "float":
                _, lo, hi, step, default = desc
                w = QDoubleSpinBox()
                w.setRange(float(lo), float(hi)); w.setSingleStep(float(step))
                w.setDecimals(6); w.setValue(float(values.get(key, default)))
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
        # collect current
        self._cache[self.agent_name] = self._collect_values()
        return self.agent_name, self._cache[self.agent_name].copy()
