import sys
from PySide6.QtWidgets import QApplication
from ui.main_section import RLLauncher


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RLLauncher()
    window.show()
    sys.exit(app.exec())
