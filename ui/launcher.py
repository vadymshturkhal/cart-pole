from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QStackedWidget
from ui.settings_dialog import SettingsDialog
from ui.training_section import TrainingSection
from ui.testing_section import TestingSection
import config


class CartPoleLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CartPole RL Launcher")
        self.resize(*config.RESOLUTION)

        # === Stack of pages ===
        self.stack = QStackedWidget()

        # Training section
        self.training_section = TrainingSection()
        self.stack.addWidget(self.training_section)
        # Connect signal
        self.training_section.back_to_main.connect(
            lambda: self.stack.setCurrentWidget(self.main_section)
        )

        # Testing section
        self.testing_section = TestingSection()
        self.stack.addWidget(self.testing_section)
        self.testing_section.back_to_main.connect(
            lambda: self.stack.setCurrentWidget(self.main_section)
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.stack)

        # Create Main and Testing sections
        self.main_section = QWidget()
        self.stack.addWidget(self.main_section)

        # Build content
        self._build_main_section()

        # Default page
        self.stack.setCurrentWidget(self.main_section)

    def _build_main_section(self):
        layout = QVBoxLayout(self.main_section)

        # === Column of big menu buttons ===
        self.training_section_btn = QPushButton("▶ Train Agent")
        self.testing_section_btn = QPushButton("🎮 Test Agent")
        self.settings_btn = QPushButton("⚙ Settings")

        # Make buttons taller/wider like a game menu
        for btn in (self.training_section_btn, self.testing_section_btn, self.settings_btn):
            btn.setMinimumHeight(50)
            btn.setStyleSheet("font-size: 18px;")  # bigger font
            layout.addWidget(btn)

        self.training_section_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.training_section))
        self.testing_section_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.testing_section))
        self.settings_btn.clicked.connect(self.open_settings)

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            values = dlg.get_values()
            config.save_user_config(values)
            if "RESOLUTION" in values:
                self.resize(*values["RESOLUTION"])
            if "EPISODES" in values:
                self.episodes_box.setValue(values["EPISODES"])
