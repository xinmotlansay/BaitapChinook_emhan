"""Cấu hình chung của dự án: đường dẫn, K, seed, siêu tham số."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "databases" / "Chinook_Sqlite.sqlite"
MODEL_PACKAGE = ROOT / "models" / "chinook_recsys.zip"
ARTIFACTS_DIR = ROOT / "artifacts"

SEED = 42
SEEDS = [42, 43, 44, 45, 46]
TOP_K_LIST = [5, 10, 20]
DEFAULT_TOP_K = 10

# Các siêu tham số Hybrid dưới đây được chọn bằng `python -m ml.tune` trên val (artifacts/tuning.csv)

# Lấy mẫu âm cho Hybrid Ranker: 1 dương : NEG_RATIO âm
NEG_RATIO = 10

# Số fold trong để dựng tập huấn luyện cho Hybrid Ranker (leave-last-invoice-out lặp lại)
HYBRID_FOLDS = 3

# Gộp điểm track -> album (tổng các phép gộp được liệt kê)
ALBUM_AGG = ("max",)

# Đánh giá
BOOTSTRAP_ROUNDS = 2000
CI_LEVEL = 0.95

# ALS (B5)
ALS_FACTORS = 32
ALS_REG = 0.1
ALS_ITERS = 15
ALS_ALPHA = 20.0

# HistGradientBoostingClassifier
GBDT_PARAMS = dict(max_iter=300, learning_rate=0.05, max_depth=6)
