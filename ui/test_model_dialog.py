import os, torch, config
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QListWidget, QPushButton, QHBoxLayout, QMessageBox

class TestModelDialog(QDialog):
    def __init__(self, folder="trained_models"):
        super().__init__()
        self.setWindowTitle("Choose Model to Test")
        self.resize(400, 300)

        self.layout = QVBoxLayout(self)

        self.label = QLabel("Select a model:")
        self.layout.addWidget(self.label)

        self.list_widget = QListWidget()
        self.models = [f for f in os.listdir(folder) if f.endswith(".pth")]

        for m in self.models:
            self.list_widget.addItem(m)

        self.layout.addWidget(self.list_widget)

        self.info_label = QLabel("Model info will appear here.")
        self.layout.addWidget(self.info_label)
        self.list_widget.currentTextChanged.connect(self.show_info)

        # Model management buttons
        btn_row = QHBoxLayout()
        self.test_btn = QPushButton("Test Selected Model")
        self.delete_btn = QPushButton("Delete Selected Model")

        btn_row.addWidget(self.test_btn)
        btn_row.addWidget(self.delete_btn)
        self.layout.addLayout(btn_row)

        self.selected_model = None
        self.test_btn.clicked.connect(self.accept)
        self.delete_btn.clicked.connect(self.delete_model)

    def show_info(self, filename):
        path = os.path.join("trained_models", filename)
        try:
            checkpoint = torch.load(path, map_location=config.DEVICE)
            if isinstance(checkpoint, dict) and "hyperparams" in checkpoint:
                hps = checkpoint["hyperparams"]
                episodes_trained = checkpoint.get("episodes_trained", "N/A")
                episodes_total = checkpoint.get("episodes_total", "N/A")
                environment = checkpoint.get("environment", "N/A")

                if episodes_total != "N/A":
                    info = f"<b>Hyperparams:</b>{hps}<br><b>Episodes: </b>{episodes_trained}/{episodes_total}<br><b>Environment: </b> {environment}"
                else:
                    info = f"<b>Hyperparams:</b>{hps}<br><b>Episodes: </b>{episodes_trained} <br><b>Environment: </b>{environment}"

            else:
                info = "⚠ Legacy model — no metadata available"

        except Exception as e:
            info = f"❌ Could not load: {e}"
        self.info_label.setText(info)
        self.selected_model = path

    def get_selected(self):
        return self.selected_model

    def delete_model(self):
        row = self.list_widget.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Warning", "⚠ No model selected")
            return

        model_file = os.path.join(
            config.TRAINED_MODELS_FOLDER,
            self.list_widget.item(row).text()
        )

        try:
            os.remove(model_file)
            self.list_widget.takeItem(row)  # remove from list
            self.info_label.setText("Model info will appear here.")
            self.selected_model = None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not delete model:\n{e}")
