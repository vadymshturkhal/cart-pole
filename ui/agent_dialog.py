from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton


class AgentDialog(QDialog):
    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Select Agent & Hyperparameters")
        self.resize(300, 400)

        layout = QVBoxLayout(self)

        # Agent list
        self.agent_list = QListWidget()
        self.agent_list.addItems(["nstep_dqn", "nstep_ddqn"])
        self.agent_list.setCurrentRow(0)
        layout.addWidget(self.agent_list)

        # Hyperparams form
        self.params = {}
        form = QFormLayout()

        self.params["gamma"] = QDoubleSpinBox(); self.params["gamma"].setValue(defaults.get("gamma", 0.99))
        form.addRow("Gamma:", self.params["gamma"])

        self.params["lr"] = QDoubleSpinBox(); self.params["lr"].setDecimals(6); self.params["lr"].setValue(defaults.get("lr", 1e-3))
        form.addRow("Learning rate:", self.params["lr"])

        self.params["buffer_size"] = QSpinBox(); self.params["buffer_size"].setMaximum(10**9); self.params["buffer_size"].setValue(defaults.get("buffer_size", 10000))
        form.addRow("Buffer size:", self.params["buffer_size"])

        self.params["batch_size"] = QSpinBox(); self.params["batch_size"].setMaximum(10**6); self.params["batch_size"].setValue(defaults.get("batch_size", 64))
        form.addRow("Batch size:", self.params["batch_size"])

        self.params["n_step"] = QSpinBox(); self.params["n_step"].setValue(defaults.get("n_step", 3))
        form.addRow("N-step:", self.params["n_step"])

        self.params["eps_start"] = QDoubleSpinBox(); self.params["eps_start"].setDecimals(2); self.params["eps_start"].setValue(defaults.get("eps_start", 1.0))
        form.addRow("Epsilon start:", self.params["eps_start"])

        self.params["eps_end"] = QDoubleSpinBox(); self.params["eps_end"].setDecimals(2); self.params["eps_end"].setValue(defaults.get("eps_end", 0.05))
        form.addRow("Epsilon end:", self.params["eps_end"])

        self.params["eps_decay"] = QSpinBox(); self.params["eps_decay"].setMaximum(10**9); self.params["eps_decay"].setValue(defaults.get("eps_decay", 10000))
        form.addRow("Epsilon decay:", self.params["eps_decay"])

        layout.addLayout(form)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)

    def get_selection(self):
        agent = self.agent_list.currentItem().text()
        hps = {k: w.value() for k, w in self.params.items()}
        return agent, hps
