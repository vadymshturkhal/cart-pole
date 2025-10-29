from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QPushButton, QGroupBox, QRadioButton,
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

        # === Other parameters (auto-added) ===
        for key, value in self.updated_params.items():
            if key in ["eps_fixed", "eps_start", "eps_end", "eps_decay", "epsilon_schedule"]:
                continue

            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(1, 1_000_000)
                widget.setValue(value)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                widget.setRange(-1_000_000.0, 1_000_000.0)
                widget.setValue(value)
            else:
                widget = QLabel(str(value))
                widget.setEnabled(False)

            display_name = "Target Network Update Interval" if key == "target_update" else key
            self.form.addRow(f"{display_name}:", widget)
            self.widgets[key] = widget

        # === Epsilon Schedule Section ===
        eps_schedule_group = QGroupBox("Epsilon Schedule Type")
        eps_schedule_layout = QHBoxLayout()

        self.eps_fixed = QRadioButton("Fixed")
        self.eps_linear = QRadioButton("Linear (episodes)")
        self.eps_exponential = QRadioButton("Exponential (episodes)")
        self.eps_manual = QRadioButton("Manual (steps)")
        eps_schedule_layout.addWidget(self.eps_fixed)
        eps_schedule_layout.addWidget(self.eps_linear)
        eps_schedule_layout.addWidget(self.eps_exponential)
        eps_schedule_layout.addWidget(self.eps_manual)
        eps_schedule_group.setLayout(eps_schedule_layout)
        self.form.addRow(eps_schedule_group)

        self.widgets["epsilon_schedule"] = [self.eps_fixed, self.eps_linear, self.eps_exponential, self.eps_manual]

        # === Epsilon Parameter Row ===
        eps_param_layout = QHBoxLayout()
        self.eps_start_spin = QDoubleSpinBox()
        self.eps_start_spin.setDecimals(4)
        self.eps_start_spin.setRange(0.0, 1.0)
        self.eps_start_spin.setValue(self.updated_params.get("eps_start", 0.8))

        self.eps_end_spin = QDoubleSpinBox()
        self.eps_end_spin.setDecimals(4)
        self.eps_end_spin.setRange(0.0, 1.0)
        self.eps_end_spin.setValue(self.updated_params.get("eps_end", 0.05))

        self.eps_decay_spin = QSpinBox()
        self.eps_decay_spin.setRange(1, 1_000_000)
        self.eps_decay_spin.setValue(self.updated_params.get("eps_decay", 10000))
        self.eps_decay_spin.setVisible(False)  # hidden initially

        eps_param_layout.addWidget(QLabel("Epsilon Start:"))
        eps_param_layout.addWidget(self.eps_start_spin)
        eps_param_layout.addWidget(QLabel("Epsilon End:"))
        eps_param_layout.addWidget(self.eps_end_spin)
        eps_param_layout.addWidget(QLabel("Epsilon Decay:"))
        eps_param_layout.addWidget(self.eps_decay_spin)
        self.form.addRow(eps_param_layout)

        self.widgets["eps_start"] = self.eps_start_spin
        self.widgets["eps_end"] = self.eps_end_spin
        self.widgets["eps_decay"] = self.eps_decay_spin

        # === Fixed epsilon single value ===
        self.eps_fixed_spin = QDoubleSpinBox()
        self.eps_fixed_spin.setDecimals(4)
        self.eps_fixed_spin.setRange(0.0, 1.0)
        self.eps_fixed_spin.setValue(self.updated_params.get("eps_fixed", 0.05))
        self.form.addRow(QLabel("Fixed epsilon:"), self.eps_fixed_spin)
        self.eps_fixed_spin.setVisible(False)
        self.widgets["eps_fixed"] = self.eps_fixed_spin
        
        # === Buttons ===
        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Close" if read_only else "Cancel")
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.save_btn.clicked.connect(self._on_apply)
        self.cancel_btn.clicked.connect(self._on_cancel)

        # === Behavior ===
        self.eps_fixed.toggled.connect(self._update_epsilon_fields)
        self.eps_linear.toggled.connect(self._update_epsilon_fields)
        self.eps_exponential.toggled.connect(self._update_epsilon_fields)
        self.eps_manual.toggled.connect(self._update_epsilon_fields)

        # Restore saved epsilon schedule
        schedule_type = self.updated_params.get("epsilon_schedule", "fixed")
        if schedule_type == "fixed":
            self.eps_fixed .setChecked(True)
        elif schedule_type == "linear":
            self.eps_linear.setChecked(True)
        elif schedule_type == "exponential":
            self.eps_exponential.setChecked(True)
        elif schedule_type == "manual":
            self.eps_manual.setChecked(True)

        # === Read-only mode ===
        if self.read_only:
            layout.addWidget(QLabel("<span style='color:#bbb;'>🔒 Read-only mode (Training in progress)</span>"))
            for w in self.widgets.values():
                if isinstance(w, list):
                    for sub_w in w:
                        sub_w.setEnabled(False)
                else:
                    w.setEnabled(False)
            self.save_btn.setEnabled(False)

    # ------------------------------------------------------------------
    def _update_epsilon_fields(self):
        """Toggle epsilon controls depending on schedule type."""
        if self.eps_fixed.isChecked():
            self.eps_fixed_spin.setVisible(True)
            self.eps_start_spin.setVisible(False)
            self.eps_end_spin.setVisible(False)
            self.eps_decay_spin.setVisible(False)
        elif self.eps_manual.isChecked():
            self.eps_fixed_spin.setVisible(False)
            self.eps_start_spin.setVisible(True)
            self.eps_end_spin.setVisible(True)
            self.eps_decay_spin.setVisible(True)
        else:  # linear or exponential
            self.eps_fixed_spin.setVisible(False)
            self.eps_start_spin.setVisible(True)
            self.eps_end_spin.setVisible(True)
            self.eps_decay_spin.setVisible(False)

    # ------------------------------------------------------------------
    def _on_apply(self):
        """Collect updated hyperparameters and return."""
        if self.eps_fixed.isChecked():
            self.updated_params["epsilon_schedule"] = "fixed"
            self.updated_params["eps_fixed"] = self.eps_fixed_spin.value()
        elif self.eps_linear.isChecked():
            self.updated_params["epsilon_schedule"] = "linear"
        elif self.eps_exponential.isChecked():
            self.updated_params["epsilon_schedule"] = "exponential"
        elif self.eps_manual.isChecked():
            self.updated_params["epsilon_schedule"] = "manual"

        self.updated_params["eps_start"] = self.eps_start_spin.value()
        self.updated_params["eps_end"] = self.eps_end_spin.value()
        self.updated_params["eps_decay"] = self.eps_decay_spin.value()

        for k, w in self.widgets.items():
            if k in ["eps_start", "eps_end", "eps_decay", "epsilon_schedule"]:
                continue
            if isinstance(w, QCheckBox):
                self.updated_params[k] = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                self.updated_params[k] = w.value()

        self.on_close_callback(True, self.updated_params)

    # ------------------------------------------------------------------
    def _on_cancel(self):
        """Close without saving."""
        self.on_close_callback(False, None)
