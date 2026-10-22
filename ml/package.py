"""Đóng gói / nạp mô hình Hybrid: models/chinook_recsys.zip (TECHSPEC mục 5).

Gồm:
  model.joblib  - bộ phân loại GBDT đã huấn luyện
  state.joblib  - phần còn lại của mô hình (thống kê, ma trận tương đồng, idf, nhân tố ALS...)
  meta.json     - phiên bản thư viện, seed, FEATURE_COLUMNS, ngày train, metric
"""
import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from config import MODEL_PACKAGE
from ml.features import FEATURE_COLUMNS
from ml.hybrid import HybridRecommender


def save_package(model, path=MODEL_PACKAGE, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": model.name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seed": model.seed,
        "feature_columns": FEATURE_COLUMNS,
        "album_agg": list(model.album_agg),
        "gbdt_params": model.gbdt_params,
        "neg_ratio": model.neg_ratio,
        "n_folds": model.n_folds,
        "train_info": model.train_info,
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__,
                      "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "metrics": metrics or {},
    }
    state = {k: v for k, v in model.__dict__.items() if k != "clf"}
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, writer in (
            ("model.joblib", lambda p: joblib.dump(model.clf, p)),
            ("state.joblib", lambda p: joblib.dump(state, p)),
            ("meta.json", lambda p: Path(p).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")),
        ):
            p = Path(tmp) / name
            writer(p)
            z.write(p, name)
    return path


def read_meta(path=MODEL_PACKAGE):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("meta.json").decode("utf-8"))


def load_package(path=MODEL_PACKAGE):
    """Nạp mô hình; từ chối nếu FEATURE_COLUMNS của gói khác với mã hiện tại."""
    meta = read_meta(path)
    if meta["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Gói mô hình được huấn luyện với FEATURE_COLUMNS khác mã hiện tại; hãy train lại "
                         "(python -m ml.train).")
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as z:
        z.extract("model.joblib", tmp)
        z.extract("state.joblib", tmp)
        clf = joblib.load(Path(tmp) / "model.joblib")
        state = joblib.load(Path(tmp) / "state.joblib")
    model = HybridRecommender.__new__(HybridRecommender)
    model.__dict__.update(state)
    model.clf = clf
    return model
