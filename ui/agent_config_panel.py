from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QPushButton,
    QDoubleSpinBox, QSpinBox, QCheckBox, QLabel, QHBoxLayout
)


class AgentConfigPanel(QWidget):
    """Inline panel for viewing and adjusting agent hyperparameters."""

    def __init__(self, agent_name: str, hyperparams: dict, on_close_callback, read_only: bool = False):
        super().__init__()
        self.agent_name = agent_name
        self.updated_params = hyperparams.copy()
        self.on_close_callback = on_close_callback
        self.read_only = read_only
        self.widgets = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel(f"<b>🤖 {agent_name} Configuration</b>")
        title.setStyleSheet("font-size:16px; color:#ddd;")
        layout.addWidget(title)

        self.form = QFormLayout()
        layout.addLayout(self.form)

        # --- Build form dynamically ---
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

        # --- Buttons ---
        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Close" if read_only else "Cancel")
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.save_btn.clicked.connect(self._on_apply)
        self.cancel_btn.clicked.connect(self._on_cancel)

        # --- Read-only mode ---
        if self.read_only:
            for w in self.widgets.values():
                w.setEnabled(False)
            self.save_btn.setEnabled(False)
            layout.addWidget(QLabel("<span style='color:#bbb;'>🔒 Read-only mode (Training in progress)</span>"))

    # ------------------------------------------------------------------
    def _on_apply(self):
        """Collect updated hyperparameters and return."""
        for k, w in self.widgets.items():
            if isinstance(w, QCheckBox):
                self.updated_params[k] = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                self.updated_params[k] = w.value()
        self.on_close_callback(True, self.updated_params)

    def _on_cancel(self):
        """Close without saving."""
        self.on_close_callback(False, None)
