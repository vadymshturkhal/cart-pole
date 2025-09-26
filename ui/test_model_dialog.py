import os, torch, config
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QListWidget, QPushButton

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

        self.test_btn = QPushButton("Test Selected Model")
        self.layout.addWidget(self.test_btn)

        self.list_widget.currentTextChanged.connect(self.show_info)

        self.selected_model = None
        self.test_btn.clicked.connect(self.accept)

    def show_info(self, filename):
        path = os.path.join("trained_models", filename)
        try:
            checkpoint = torch.load(path, map_location=config.DEVICE)
            if isinstance(checkpoint, dict) and "hyperparams" in checkpoint:
                hps = checkpoint["hyperparams"]
                episodes_trained = checkpoint.get("episodes_trained", "N/A")
                episodes_total = checkpoint.get("episodes_total", "N/A")

                if episodes_total != "N/A":
                    info = f"<b>Hyperparams:</b> {hps}<br><b>Episodes:</b> {episodes_trained}/{episodes_total}"
                else:
                    info = f"<b>Hyperparams:</b> {hps}<br><b>Episodes:</b> {episodes_trained}"
            else:
                info = "⚠ Legacy model — no metadata available"

        except Exception as e:
            info = f"❌ Could not load: {e}"
        self.info_label.setText(info)
        self.selected_model = path

    def get_selected(self):
        return self.selected_model
