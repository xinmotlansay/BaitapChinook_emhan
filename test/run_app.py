"""Điểm chạy UI: chuột phải -> Run trong PyCharm."""
import sys
from pathlib import Path

# Thêm project root vào sys.path để chạy được mà không cần cấu hình thêm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from ui.MainWindowEx import MainWindowEx  # noqa: E402


def main():
    app = QApplication(sys.argv)
    window = MainWindowEx()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
