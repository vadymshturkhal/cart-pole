from PySide6.QtWidgets import QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QComboBox, QSpinBox
from ui.reward_plot import RewardPlot
import config


class TrainingSection():
   def build_training_page(gui):
        layout = QVBoxLayout(gui.train_page)

        # Plot
        gui.plot = RewardPlot()
        layout.addWidget(gui.plot)

        layout.addWidget(QLabel("Environment:"))
        gui.env_box = QComboBox()
        gui.env_box.addItems(config.AVAILABLE_ENVIRONMENTS)
        gui.env_box.setCurrentText(config.DEFAULT_ENVIRONMENT)
        layout.addWidget(gui.env_box)

        layout.addWidget(QLabel("Agent:"))
        gui.agent_btn = QPushButton("Choose Agent")
        gui.train_btn = QPushButton("Start Training")
        gui.stop_btn = QPushButton("Stop Training")
        gui.save_btn = QPushButton("Save Model")
        row = QHBoxLayout()
        row.addWidget(gui.agent_btn)
        row.addWidget(gui.train_btn)
        row.addWidget(gui.stop_btn)
        row.addWidget(gui.save_btn)
        layout.addLayout(row)

        layout.addWidget(QLabel("Rendering Mode:"))
        gui.render_box = QComboBox()
        gui.render_box.addItems(["off", "human", "gif", "mp4"])
        layout.addWidget(gui.render_box)

        layout.addWidget(QLabel("Training Episodes:"))
        gui.episodes_box = QSpinBox()
        gui.episodes_box.setRange(100, 10000)
        gui.episodes_box.setValue(config.EPISODES)
        layout.addWidget(gui.episodes_box)

        gui.status_label = QLabel("Idle")
        layout.addWidget(gui.status_label)

        # Back to main menu
        back_btn = QPushButton("⬅ Back to Main Menu")
        back_btn.setMinimumHeight(40)
        back_btn.setStyleSheet("font-size: 16px;")
        back_btn.clicked.connect(lambda: gui.stack.setCurrentWidget(gui.main_page))
        back_btn.clicked.connect(gui.stop_viewer_and_back)
        layout.addWidget(back_btn)

        # Connects
        gui.agent_btn.clicked.connect(gui.choose_agent)
        gui.train_btn.clicked.connect(gui.start_training)
        gui.stop_btn.clicked.connect(gui.stop_training)
        gui.save_btn.clicked.connect(gui.save_agent_as)
