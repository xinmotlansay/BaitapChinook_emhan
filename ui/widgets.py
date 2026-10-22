"""Widget dùng lại: bảng pandas có sắp xếp, canvas matplotlib theo palette của hệ điều hành."""
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QAbstractTableModel, QSortFilterProxyModel, Qt
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableView, QVBoxLayout, QWidget


class PandasModel(QAbstractTableModel):
    """Hiển thị DataFrame. `raw` (tùy chọn) là DataFrame cùng hình dạng dùng để sắp xếp theo số."""

    def __init__(self, df=None, raw=None):
        super().__init__()
        self.set_data(df if df is not None else pd.DataFrame(), raw)

    def set_data(self, df, raw=None):
        self.beginResetModel()
        self.df = df.reset_index(drop=True)
        self.raw = (raw if raw is not None else df).reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=None):
        return len(self.df)

    def columnCount(self, parent=None):
        return len(self.df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{value:.4f}" if isinstance(value, float) else str(value)
        if role == Qt.ItemDataRole.ToolTipRole:  # rê chuột để đọc đủ chữ bị cắt
            return str(value)
        if role == Qt.ItemDataRole.UserRole:  # giá trị dùng để sắp xếp
            return self.raw.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (int, float)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return str(self.df.columns[section])
        return None


class SortableTable:
    """Gắn PandasModel + proxy sắp xếp vào một QTableView có sẵn từ file .ui."""

    def __init__(self, view: QTableView, stretch_last=True):
        self.model = PandasModel()
        self.proxy = QSortFilterProxyModel(view)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.ItemDataRole.UserRole)
        view.setModel(self.proxy)
        view.setSortingEnabled(True)
        # không sắp xếp sẵn: giữ thứ tự gốc (xếp hạng, ngày mới nhất trước) tới khi người dùng bấm tiêu đề
        view.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        view.setAlternatingRowColors(True)
        view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        view.verticalHeader().setVisible(False)
        view.horizontalHeader().setStretchLastSection(stretch_last)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.view = view

    def set_data(self, df, raw=None):
        self.model.set_data(df, raw)
        self.view.resizeColumnsToContents()
        # cột chữ dài (tên bài, lý do) không nên chiếm quá nửa bảng
        for c in range(self.model.columnCount()):
            self.view.setColumnWidth(c, min(self.view.columnWidth(c), 220))


class MplCanvas(FigureCanvasQTAgg):
    """Canvas matplotlib; màu nền/chữ lấy theo palette hiện tại để hợp chế độ sáng/tối."""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(5, 3), layout="constrained")
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)

    def _colors(self):
        pal = self.palette()
        return pal.window().color().name(), pal.windowText().color().name(), pal.mid().color().name()

    def reset(self):
        """Xóa hình và vẽ lại trục trống đã áp màu theo palette."""
        bg, fg, grid = self._colors()
        self.figure.clear()
        self.figure.set_facecolor(bg)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(bg)
        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            self.ax.spines[spine].set_color(grid)
        self.ax.tick_params(colors=fg)
        self.ax.xaxis.label.set_color(fg)
        self.ax.yaxis.label.set_color(fg)
        self.ax.title.set_color(fg)
        return self.ax

    def message(self, text):
        ax = self.reset()
        ax.set_axis_off()
        ax.text(0.5, 0.5, text, ha="center", va="center", color=self._colors()[1], transform=ax.transAxes)
        self.draw_idle()


def attach_canvas(host: QWidget) -> MplCanvas:
    """Đặt một MplCanvas vào QWidget trống của file .ui."""
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    canvas = MplCanvas(host)
    layout.addWidget(canvas)
    return canvas
