from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QPushButton, QDoubleSpinBox,
    QSpinBox, QCheckBox, QLabel
)

class AgentDetailsDialog(QDialog):
    """
        Editable dialog for viewing and adjusting agent hyperparameters.
    """

    def __init__(self, agent_name: str, hyperparams: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Agent Hyperparameters — {agent_name}")
        self.resize(420, 520)
        self.agent_name = agent_name
        self._original_params = hyperparams
        self.updated_params = hyperparams.copy()

        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        layout.addLayout(self.form)

        # --- Build form dynamically ---
        self.widgets = {}
        for key, value in self.updated_params.items():
            widget = None
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(1, 1_000_000)
                widget.setValue(value)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setDecimals(6)
                widget.setRange(-1_000_000.0, 1_000_000.0)
                widget.setValue(value)
            else:
                widget = QLabel(str(value))
                widget.setEnabled(False)
            self.form.addRow(f"{key}:", widget)
            self.widgets[key] = widget

        # Buttons
        btn_layout = QVBoxLayout()
        save_btn = QPushButton("Save Changes")
        save_btn.clicked.connect(self._on_save)
        close_btn = QPushButton("Cancel")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _on_save(self):
        """Collect updated hyperparameters."""
        for k, w in self.widgets.items():
            if isinstance(w, QCheckBox):
                self.updated_params[k] = w.isChecked()
            elif isinstance(w, QSpinBox):
                self.updated_params[k] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                self.updated_params[k] = w.value()
            # Non-editable QLabel values stay as is
        self.accept()

    def get_updated_params(self) -> dict:
        return self.updated_params
