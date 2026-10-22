"""Dự đoán: nạp gói mô hình và trả Top-K gợi ý cho một khách.

CLI : python -m ml.predict --customer 5 --kind track --k 10
UI  : from ml.predict import recommend, list_models

Mô hình chính (M_hybrid) nạp từ models/chinook_recsys.zip. Các baseline B0-B5 được fit
tại chỗ trên train+val (cùng dữ liệu với gói) để UI so sánh; kết quả được cache trong tiến trình.
"""
import argparse
from functools import lru_cache

import pandas as pd

from config import DEFAULT_TOP_K, MODEL_PACKAGE, SEED
from dal.CustomerDAO import CustomerDAO
from ml.data_prep import split_by_time
from ml.hybrid import HybridRecommender
from ml.package import load_package, read_meta
from ml.recommenders import (ALSRecommender, ContentBasedRecommender, GenreArtistPopRecommender,
                             ItemKNNRecommender, PopularityRecommender, RandomRecommender)

KINDS = ("track", "album")
BASELINES = {cls.name: cls for cls in (
    RandomRecommender, PopularityRecommender, GenreArtistPopRecommender,
    ItemKNNRecommender, ContentBasedRecommender, ALSRecommender)}
MAIN_MODEL = HybridRecommender.name


def list_models():
    return [MAIN_MODEL] + list(BASELINES)


@lru_cache(maxsize=1)
def _main_model():
    if not MODEL_PACKAGE.exists():
        raise FileNotFoundError(f"Chưa có {MODEL_PACKAGE.name}. Hãy chạy: python -m ml.train")
    return load_package(MODEL_PACKAGE)


@lru_cache(maxsize=None)
def get_model(name=MAIN_MODEL):
    if name == MAIN_MODEL:
        return _main_model()
    if name not in BASELINES:
        raise ValueError(f"Mô hình không hợp lệ: {name}. Chọn một trong {list_models()}")
    split = split_by_time()
    return BASELINES[name](SEED).fit(split.train_plus_val(), split.tracks, CustomerDAO.get_all())


def model_meta():
    """Metadata của gói mô hình chính (cho tab thông tin trên UI)."""
    return read_meta(MODEL_PACKAGE)


def recommend(customer_id, kind="track", k=DEFAULT_TOP_K, model=MAIN_MODEL):
    """Trả list[Recommendation] (item_id, name, artist_name, score, reason), xếp hạng giảm dần."""
    if kind not in KINDS:
        raise ValueError(f"kind phải là một trong {KINDS}")
    if k < 1:
        raise ValueError("k phải >= 1")
    m = get_model(model)
    if customer_id not in m.state.country_of:
        raise ValueError(f"Không tìm thấy khách hàng có mã {customer_id}")
    return m.recommend_tracks(customer_id, k) if kind == "track" else m.recommend_albums(customer_id, k)


SIGNAL_COLUMNS = ["pop_track", "artist_affinity", "genre_affinity", "country_genre_pop",
                  "knn_score", "content_cosine", "mf_score"]


def explain(customer_id, item_id, kind="track", model=MAIN_MODEL):
    """Chi tiết cho một gợi ý: thông tin item, đối chiếu với lịch sử mua và (Hybrid/track) các tín hiệu mô hình.

    Trả về dict: album, artist, genre, n_tracks, bought_total, same_artist, same_genre, same_album,
    signals (dict hoặc None).
    """
    m = get_model(model)
    t = m.state.tracks
    rows = t.loc[[item_id]] if kind == "track" else t[t.album_id == item_id]
    first = rows.iloc[0]
    genre_id = rows.genre_id.mode().iloc[0]
    hist = m.state.hist_by_customer.get(customer_id)
    if hist is None:
        hist = pd.DataFrame({"album_id": [], "artist_id": [], "genre_id": []})
    info = {
        "album": first.album_title, "artist": first.artist_name,
        "genre": rows.genre_name.mode().iloc[0], "n_tracks": len(rows),
        "bought_total": len(hist),
        "same_artist": int((hist.artist_id == first.artist_id).sum()),
        "same_genre": int((hist.genre_id == genre_id).sum()),
        "same_album": int((hist.album_id == first.album_id).sum()),
        "signals": None,
    }
    if model == MAIN_MODEL and kind == "track" and m.state.has_history(customer_id):
        f = m.comp.features(customer_id, pd.Index([item_id])).iloc[0]
        info["signals"] = {c: float(f[c]) for c in SIGNAL_COLUMNS}
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gợi ý Top-K Track/Album cho một khách hàng Chinook")
    ap.add_argument("--customer", type=int, required=True, help="CustomerId")
    ap.add_argument("--kind", choices=KINDS, default="track")
    ap.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--model", choices=list_models(), default=MAIN_MODEL)
    args = ap.parse_args(argv)

    try:
        recs = recommend(args.customer, args.kind, args.k, args.model)
    except (ValueError, FileNotFoundError) as e:
        ap.exit(1, f"Lỗi: {e}\n")

    m = get_model(args.model)
    n_bought = len(m.state.bought_tracks(args.customer))
    print(f"Khách {args.customer} ({m.state.country_of[args.customer]}, đã mua {n_bought} track) "
          f"— Top-{args.k} {args.kind} bởi {args.model}")
    for i, r in enumerate(recs, 1):
        print(f"{i:>2}. {r.name} — {r.artist_name}  [điểm {r.score:.4f}]  {r.reason}")


if __name__ == "__main__":
    main()
