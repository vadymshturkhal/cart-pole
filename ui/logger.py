from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QFileDialog
)
from PySide6.QtCore import Qt


class EpisodeLogWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)

        self.table = QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(
            ["Episode", "Reward", "Avg20", "GlobalAvg"]
        )
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.table.setSortingEnabled(True)
        self.table.setMaximumHeight(220)
        v.addWidget(self.table)

        h = QHBoxLayout()
        self.btn_clear = QPushButton("Clear Log")
        self.btn_export = QPushButton("Export CSV")
        h.addWidget(self.btn_clear)
        h.addWidget(self.btn_export)
        v.addLayout(h)

        self.btn_clear.clicked.connect(self.clear)
        self.btn_export.clicked.connect(self.export_csv)

    def append_row(self, ep:int, reward:float, avg20:float, global_avg:float,
                   steps:int|None=None, epsilon:float|None=None):
        r = self.table.rowCount()
        self.table.insertRow(r)

        vals = [
            ep + 1,
            round(reward, 3),
            round(avg20, 3),
            round(global_avg, 3),
            None if steps is None else steps,
            None if epsilon is None else round(epsilon, 6),
        ]
        for c, v in enumerate(vals):
            itm = QTableWidgetItem("" if v is None else str(v))
            # numeric sorting via DisplayRole
            if v is not None:
                itm.setData(Qt.ItemDataRole.DisplayRole, v)
            self.table.setItem(r, c, itm)

        self.table.scrollToBottom()

    def clear(self):
        self.table.setRowCount(0)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save log as CSV", "training_log.csv", "CSV (*.csv)"
        )
        if not path:
            return
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            headers = [self.table.horizontalHeaderItem(c).text()
                       for c in range(self.table.columnCount())]
            w.writerow(headers)
            for r in range(self.table.rowCount()):
                w.writerow([
                    self.table.item(r, c).text() if self.table.item(r, c) else ""
                    for c in range(self.table.columnCount())
                ])
