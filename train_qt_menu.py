import sys
from PySide6.QtWidgets import QApplication
from ui.launcher import CartPoleLauncher


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CartPoleLauncher()
    window.show()
    sys.exit(app.exec())
