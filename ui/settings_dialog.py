# ui/settings_dialog.py
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QComboBox, QSpinBox,
    QPushButton, QHBoxLayout
)
import config

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)

        # ===== Resolution options =====
        layout.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        self.resolutions = {
            "640 x 480 (VGA)": [640, 480],
            "800 x 600 (SVGA)": [800, 600],
            "1280 x 720 (HD)": [1280, 720],
            "1366 x 768 (WXGA)": [1366, 768],
            "1600 x 900 (HD+)": [1600, 900],
            "1920 x 1080 (Full HD)": [1920, 1080],
        }
        self.res_combo.addItems(self.resolutions.keys())

        # pre-select current resolution
        current_res = config.RESOLUTION
        for name, res in self.resolutions.items():
            if res == current_res:
                self.res_combo.setCurrentText(name)
                break

        layout.addWidget(self.res_combo)

        # ===== Buttons =====
        btns = QHBoxLayout()
        self.ok_btn = QPushButton("Save")
        self.cancel_btn = QPushButton("Cancel")
        btns.addWidget(self.ok_btn)
        btns.addWidget(self.cancel_btn)
        layout.addLayout(btns)

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def get_values(self):
        return {
            "RESOLUTION": self.resolutions[self.res_combo.currentText()],
        }
