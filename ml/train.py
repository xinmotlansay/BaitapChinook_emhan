"""Huấn luyện + đánh giá cuối trên test + sinh artifacts + đóng gói.

Chạy: python -m ml.train

Quy trình (TECHSPEC mục 4, 6, 9):
  - Mọi mô hình fit trên train+val, đo trên test.
  - Với mỗi seed trong SEEDS: B0, B5 và Hybrid thay đổi theo seed (lấy mẫu âm, khởi tạo MF).
  - metrics.csv / paired.csv: kết quả của seed đầu tiên (SEED), có CI bootstrap 95 %.
  - seeds.csv: trung bình ± độ lệch chuẩn của `mean` qua các seed.
  - Siêu tham số Hybrid lấy từ config.py (đã chọn bằng ml.tune trên val).
"""
import time

from joblib import Parallel, delayed

from config import ARTIFACTS_DIR, MODEL_PACKAGE, SEED, SEEDS
from dal.CustomerDAO import CustomerDAO
from ml.data_prep import split_by_time
from ml.evaluate import aggregate_seeds, evaluate_models
from ml.hybrid import HybridRecommender
from ml.package import load_package, save_package
from ml.recommenders import (ALSRecommender, ContentBasedRecommender, GenreArtistPopRecommender,
                             ItemKNNRecommender, PopularityRecommender, RandomRecommender)

MAIN = HybridRecommender.name


def build_models(seed):
    return [RandomRecommender(seed), PopularityRecommender(seed), GenreArtistPopRecommender(seed),
            ItemKNNRecommender(seed), ContentBasedRecommender(seed), ALSRecommender(seed),
            HybridRecommender(seed)]


def run_seed(seed, split, customers):
    return evaluate_models(build_models(seed), split, which="test", main=MAIN,
                           customers=customers, seed=seed)


def main():
    t0 = time.time()
    split, customers = split_by_time(), CustomerDAO.get_all()
    print(f"Dữ liệu: train+val={len(split.train_plus_val())} dòng, test={len(split.test)} dòng, "
          f"{split.test.customer_id.nunique()} khách")

    results = Parallel(n_jobs=min(len(SEEDS), 8))(delayed(run_seed)(s, split, customers) for s in SEEDS)
    by_seed = dict(zip(SEEDS, results))
    metrics, paired = by_seed[SEED]

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)
    paired.to_csv(ARTIFACTS_DIR / "paired.csv", index=False)
    aggregate_seeds({s: m for s, (m, _) in by_seed.items()}).to_csv(ARTIFACTS_DIR / "seeds.csv", index=False)
    print(f"Đã ghi metrics.csv, paired.csv, seeds.csv vào {ARTIFACTS_DIR}")

    # Gói mô hình cuối: Hybrid fit trên train+val
    final = HybridRecommender(SEED).fit(split.train_plus_val(), split.tracks, customers)
    summary = metrics[(metrics.model == MAIN) & (metrics.group == "all") & (metrics.k == 10)]
    save_package(final, MODEL_PACKAGE, metrics={
        f"{r.level}_{r.metric}@10": {"mean": r.mean, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi}
        for r in summary.itertuples() if r.metric in ("recall", "hitrate", "ndcg", "map")
    })

    # Kiểm tra gói nạp lại cho đúng kết quả cũ
    loaded = load_package(MODEL_PACKAGE)
    cust = sorted(set(split.train_plus_val().customer_id))[0]
    a = [r.item_id for r in final.recommend_tracks(cust, 10)]
    b = [r.item_id for r in loaded.recommend_tracks(cust, 10)]
    assert a == b, "Gói mô hình nạp lại cho kết quả khác mô hình gốc"
    print(f"Đã lưu {MODEL_PACKAGE} (đã kiểm tra nạp lại). Tổng thời gian {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
