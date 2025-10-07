from PySide6.QtWidgets import (
    QVBoxLayout, QPushButton, QDialog, QTreeWidget, 
    QTreeWidgetItem, QHeaderView, QVBoxLayout
)

class AgentDetailsDialog(QDialog):
    def __init__(self, agent_name: str, hyperparams: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Agent Details — {agent_name}")
        self.resize(400, 500)

        layout = QVBoxLayout(self)

        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Parameter", "Value"])
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        tree.setAlternatingRowColors(True)
        tree.setMinimumHeight(400)
        layout.addWidget(tree)

        if isinstance(hyperparams, dict):
            for k in sorted(hyperparams.keys()):
                QTreeWidgetItem(tree, [str(k), str(hyperparams[k])])
            tree.expandAll()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)