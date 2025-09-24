from PySide6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QPushButton, QDoubleSpinBox, QSpinBox

class HyperparamsDialog(QDialog):
    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Adjust Hyperparameters")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Spinboxes
        self.gamma_box = QDoubleSpinBox(); self.gamma_box.setDecimals(4); self.gamma_box.setValue(defaults.get("gamma", 0.99)); form.addRow("Gamma:", self.gamma_box)
        self.lr_box = QDoubleSpinBox(); self.lr_box.setDecimals(6); self.lr_box.setValue(defaults.get("lr", 1e-3)); form.addRow("Learning Rate:", self.lr_box)
        self.buffer_box = QSpinBox(); self.buffer_box.setMaximum(10**9); self.buffer_box.setValue(defaults.get("buffer_size", 10000)); form.addRow("Buffer Size:", self.buffer_box)
        self.batch_box = QSpinBox(); self.batch_box.setMaximum(10**6); self.batch_box.setValue(defaults.get("batch_size", 64)); form.addRow("Batch Size:", self.batch_box)
        self.nstep_box = QSpinBox(); self.nstep_box.setMaximum(100); self.nstep_box.setValue(defaults.get("n_step", 3)); form.addRow("N-step:", self.nstep_box)
        self.eps_start_box = QDoubleSpinBox(); self.eps_start_box.setDecimals(3); self.eps_start_box.setValue(defaults.get("eps_start", 1.0)); form.addRow("Epsilon Start:", self.eps_start_box)
        self.eps_end_box = QDoubleSpinBox(); self.eps_end_box.setDecimals(3); self.eps_end_box.setValue(defaults.get("eps_end", 0.05)); form.addRow("Epsilon End:", self.eps_end_box)
        self.eps_decay_box = QSpinBox(); self.eps_decay_box.setMaximum(10**9); self.eps_decay_box.setValue(defaults.get("eps_decay", 10000)); form.addRow("Epsilon Decay:", self.eps_decay_box)

        layout.addLayout(form)

        # Save button
        btn = QPushButton("Save")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def get_params(self):
        return {
            "gamma": self.gamma_box.value(),
            "lr": self.lr_box.value(),
            "buffer_size": self.buffer_box.value(),
            "batch_size": self.batch_box.value(),
            "n_step": self.nstep_box.value(),
            "eps_start": self.eps_start_box.value(),
            "eps_end": self.eps_end_box.value(),
            "eps_decay": self.eps_decay_box.value(),
        }
