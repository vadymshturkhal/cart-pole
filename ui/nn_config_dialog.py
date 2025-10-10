from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox,
    QComboBox, QPushButton, QLabel, QMessageBox
)
from PySide6.QtCore import Qt
import config


class NNConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Neural Network")
        self.resize(420, 380)

        self.updated_config = {
            "HIDDEN_LAYERS": config.HIDDEN_LAYERS.copy(),
            "LR": config.LR,
            "ACTIVATION": config.ACTIVATION,
            "DROPOUT": config.DROPOUT,
            "DEVICE": str(config.DEVICE),
        }

        layout = QVBoxLayout(self)
        header = QLabel("<b>Neural Network Configuration</b>")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

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

        # Activation
        self.activation_box = QComboBox()
        self.activation_box.addItems(["relu", "tanh", "sigmoid"])
        self.activation_box.setCurrentText(config.ACTIVATION)
        form.addRow("Activation Function:", self.activation_box)

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

        # Device Selection
        self.device_box = QComboBox()
        self.device_box.addItems(["auto", "cpu", "cuda"])
        # Match current device
        current_device = "cuda" if "cuda" in str(config.DEVICE) else "cpu"
        self.device_box.setCurrentText(current_device)
        form.addRow("Computation Device:", self.device_box)

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

    def _collect_updates(self):
        """Gather values from form widgets"""
        # Architecture
        self.updated_config["HIDDEN_LAYERS"] = eval(self.hidden_layers_input.currentText())
        self.updated_config["ACTIVATION"] = self.activation_box.currentText()
        self.updated_config["DROPOUT"] = self.dropout_spin.value()

        # Optimization
        self.updated_config["LR"] = self.lr_spin.value()
        self.updated_config["OPTIMIZER"] = self.optimizer_box.currentText()

        # Device
        dev_choice = self.device_box.currentText()
        if dev_choice == "auto":
            self.updated_config["DEVICE"] = "cuda" if config.torch.cuda.is_available() else "cpu"
        else:
            self.updated_config["DEVICE"] = dev_choice

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

        # Rebuild DEVICE as torch.device (so PyTorch actually uses it)
        if "DEVICE" in self.updated_config:
            config.DEVICE = config.torch.device(self.updated_config["DEVICE"])
            