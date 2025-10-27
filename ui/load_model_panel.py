from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QListWidget, QPushButton, QHBoxLayout
)
import os
import datetime
import config


class LoadModelPanel(QWidget):
    """Panel for selecting a model file (temporarily replaces plots)."""

    def __init__(self, on_select_callback):
        super().__init__()
        self.on_select_callback = on_select_callback
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)

        title = QLabel("<b>📂 Select Model to Load</b>")
        title.setStyleSheet("font-size:16px; color:#ddd;")
        self.layout.addWidget(title)

        self.model_list = QListWidget()
        self.layout.addWidget(self.model_list)

        # Buttons
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Selected")
        self.cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.cancel_btn)
        self.layout.addLayout(btn_row)

        self.load_btn.clicked.connect(self._on_load)
        self.cancel_btn.clicked.connect(self._on_cancel)

        self._populate_model_list()

    def _populate_model_list(self):
        folder = config.TRAINED_MODELS_FOLDER
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Recursive search
        model_paths = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".pth"):
                    full_path = os.path.join(root, file)
                    model_paths.append(full_path)

        model_paths.sort()
        
        for path in model_paths:
            file = os.path.relpath(path, config.TRAINED_MODELS_FOLDER)
            t = os.path.getmtime(path)
            time_str = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
            self.model_list.addItem(f"{file}  ({time_str})")

    def _on_load(self):
        item = self.model_list.currentItem()
        if not item:
            return
        file = item.text().split("  (")[0]
        full_path = os.path.join(config.TRAINED_MODELS_FOLDER, file)
        self.on_select_callback(full_path)

    def _on_cancel(self):
        self.on_select_callback(None)
