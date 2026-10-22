"""Hàm tiện ích cho UI: định dạng, hộp thoại lỗi, chạy tác vụ nền."""
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

MODEL_LABELS = {
    "M_hybrid": "M — Hybrid Ranker (mô hình chính)",
    "B0_random": "B0 — Random",
    "B1_popularity": "B1 — Popularity",
    "B2_genre_artist_pop": "B2 — Thể loại/Nghệ sĩ × phổ biến",
    "B3_item_knn": "B3 — Item-kNN (album)",
    "B4_content": "B4 — Content-based",
    "B5_als": "B5 — Matrix Factorization (ALS)",
}
METRIC_LABELS = {
    "recall": "Recall", "hitrate": "HitRate", "ndcg": "NDCG", "map": "MAP",
    "coverage": "Coverage (chẩn đoán)", "novelty": "Novelty (chẩn đoán)",
}
GROUP_LABELS = {
    "all": "Tất cả", "seen": "Item đã thấy trong train", "cold": "Item cold (chưa ai mua)",
}
LEVEL_LABELS = {"track": "Track", "album": "Album"}
# (nhãn, kiểu hiển thị) cho các tín hiệu của Hybrid
SIGNAL_LABELS = {
    "pop_track": ("Số khách khác đã mua track này", "int"),
    "artist_affinity": ("Tỷ lệ lượt mua của bạn cùng nghệ sĩ", "pct"),
    "genre_affinity": ("Tỷ lệ lượt mua của bạn cùng thể loại", "pct"),
    "country_genre_pop": ("Mức ưa chuộng thể loại này ở quốc gia của bạn", "pct"),
    "knn_score": ("Điểm đồng mua (Item-kNN)", "num"),
    "content_cosine": ("Độ giống nội dung với hồ sơ của bạn (cosine)", "num"),
    "mf_score": ("Điểm Matrix Factorization (ALS)", "num"),
}


def model_label(model_id):
    return MODEL_LABELS.get(model_id, model_id)


def fmt_date(value):
    """Ngày dạng dd/mm/yyyy; giá trị rỗng -> chuỗi rỗng."""
    try:
        return value.strftime("%d/%m/%Y")
    except AttributeError:
        return "" if value is None else str(value)


def fmt_ci(mean, lo, hi):
    """'0.0731 [0.0434–0.1052]'; không có CI (coverage/novelty) thì chỉ in trung bình."""
    if mean != mean:  # NaN
        return "—"
    if lo != lo or hi != hi:
        return f"{mean:.4f}"
    return f"{mean:.4f} [{lo:.4f}–{hi:.4f}]"


def show_error(parent, message, title="Lỗi"):
    QMessageBox.critical(parent, title, message)


class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)


class Worker(QRunnable):
    """Chạy `fn()` trên thread pool để UI không bị đơ; kết quả/lỗi trả về qua signal."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn()
        except Exception as e:  # hiển thị cho người dùng thay vì làm sập tiến trình
            self.signals.failed.emit(str(e))
        else:
            self.signals.finished.emit(result)
