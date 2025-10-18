from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox,
    QComboBox, QPushButton, QLabel, QMessageBox
)
from PySide6.QtCore import Qt
import config
import torch


class NNConfigDialog(QDialog):
    def __init__(self, parent=None,  read_only: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Neural Network Configuration")
        self.resize(420, 380)
        self.read_only = read_only

        self.updated_config = {
            "HIDDEN_LAYERS": config.HIDDEN_LAYERS.copy(),
            "LR": config.LR,
            "ACTIVATION": config.ACTIVATION,
            "DROPOUT": config.DROPOUT,
            "DEVICE": str(config.DEVICE),
        }

        layout = QVBoxLayout(self)

        # Architecture section
        arch_label = QLabel("<b>🧩 Architecture</b>")
        arch_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(arch_label)

        form = QFormLayout()
        layout.addLayout(form)

        # Hidden Layers
        self.hidden_layers_input = QComboBox()
        presets = ["[64, 64]", "[128, 128]", "[256, 128]", "[256, 256]"]
        self.hidden_layers_input.addItems(presets)
        current = str(config.HIDDEN_LAYERS)
        if current in presets:
            self.hidden_layers_input.setCurrentText(current)
        form.addRow("Hidden Layers:", self.hidden_layers_input)

        # Hidden layers activation
        self.hidden_activation_box = QComboBox()
        self.hidden_activation_box.addItems(config.HIDDEN_ACTIVATIONS)
        self.hidden_activation_box.setCurrentText(config.HIDDEN_ACTIVATION)
        form.addRow("Hidden Activation:", self.hidden_activation_box)

        # Dropout
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(config.DROPOUT)
        form.addRow("Dropout Rate:", self.dropout_spin)

        # Optimization section
        opt_label = QLabel("<b>⚙️ Optimization</b>")
        opt_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(opt_label)

        form = QFormLayout()
        layout.addLayout(form)

        # Learning Rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(config.LR)
        form.addRow("Learning Rate:", self.lr_spin)

        # Optimizer
        self.optimizer_box = QComboBox()
        self.optimizer_box.addItems(["adam", "rmsprop", "sgd"])
        self.optimizer_box.setCurrentText(getattr(config, "OPTIMIZER", "adam"))
        form.addRow("Optimizer:", self.optimizer_box)

        # Buttons
        save_btn = QPushButton("Apply")
        save_btn.clicked.connect(self._on_apply)

        save_default_btn = QPushButton("Save as Default")
        save_default_btn.clicked.connect(self._on_save_default)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        layout.addWidget(save_btn)
        layout.addWidget(save_default_btn)
        layout.addWidget(cancel_btn)

        # --- Disable editing in read-only mode ---
        if self.read_only:
            for w in [
                self.hidden_layers_input, self.hidden_activation_box, self.dropout_spin,
                self.lr_spin, self.optimizer_box
            ]:
                w.setEnabled(False)
            save_btn.setEnabled(False)
            save_default_btn.setEnabled(False)
            layout.addWidget(QLabel("<span style='color:#bbb;'>🔒 Read-only mode (Training in progress)</span>"))

    def _collect_updates(self):
        """Gather values from form widgets"""
        # Architecture
        self.updated_config["HIDDEN_LAYERS"] = eval(self.hidden_layers_input.currentText())
        self.updated_config["HIDDEN_ACTIVATION"] = self.hidden_activation_box.currentText()
        self.updated_config["DROPOUT"] = self.dropout_spin.value()

        # Optimization
        self.updated_config["LR"] = self.lr_spin.value()
        self.updated_config["OPTIMIZER"] = self.optimizer_box.currentText()

    # Runtime apply
    def _on_apply(self):
        self._collect_updates()
        self._apply_to_runtime_config()
        self.accept()

    def _on_save_default(self):
        self._collect_updates()
        self._apply_to_runtime_config()
        config.save_user_config(self.updated_config)
        QMessageBox.information(self, "Saved", "✅ Neural network configuration saved as default.")
        self.accept()

    def get_updated_config(self):
        return self.updated_config
    
    def _apply_to_runtime_config(self):
        """Apply updated values to the runtime config module."""
        for key, value in self.updated_config.items():
            setattr(config, key, value)

    def _on_device_changed(self, choice: str):
        """Handle user selection of computation device."""

        if choice == "auto":
            config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            config.DEVICE = torch.device(choice)

        # Optional
        color = "#3a7" if "cuda" in str(config.DEVICE) else "#666"
        self.device_label.setStyleSheet(f"font-weight:bold; color:{color}; margin-left:10px;")
        self.status_label.setText(f"🖥️ Computation device set to {config.DEVICE}")
