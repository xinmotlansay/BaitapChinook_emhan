"""Chọn siêu tham số Hybrid Ranker trên tập val (fit trên train). Tuyệt đối không đụng tới test.

Chạy: python -m ml.tune   ->  artifacts/tuning.csv

Tiêu chí chọn: NDCG@10 mức album trung bình qua các seed (mức track quá thưa để làm tiêu chí).
Cả 59 khách đều thuộc val nên cấu hình tốt nhất có "winner's curse"; xem cột std_over_seeds
và so với baseline khi đọc kết quả.
"""
import itertools

import pandas as pd
from joblib import Parallel, delayed

from config import ARTIFACTS_DIR, SEEDS
from dal.CustomerDAO import CustomerDAO
from ml.data_prep import split_by_time
from ml.evaluate import get_rankings, per_customer_metrics, relevant_sets
from ml.hybrid import HybridRecommender

TUNE_SEEDS = SEEDS[:3]
K = 10
ALBUM_AGGS = [("max",), ("mean",), ("max", "mean")]
GRID = dict(
    neg_ratio=[3, 5, 10],
    n_folds=[2, 3],  # train mỗi khách có >=4 hóa đơn nên tối đa 3 fold trong
    max_depth=[3, 6],
    lr_iter=[(0.05, 100), (0.05, 300)],
)


def run_one(params, seed, split, customers):
    gbdt = dict(max_depth=params["max_depth"], learning_rate=params["lr_iter"][0],
                max_iter=params["lr_iter"][1])
    model = HybridRecommender(seed=seed, neg_ratio=params["neg_ratio"], n_folds=params["n_folds"],
                              gbdt_params=gbdt).fit(split.train, split.tracks, customers)
    custs = sorted(set(split.train.customer_id))
    rel = {lv: relevant_sets(split.val, split.tracks, split.train, lv, "all") for lv in ("track", "album")}

    track = per_customer_metrics(get_rankings(model, custs, "track", K), rel["track"], [K]).mean()
    rows = []
    for agg in ALBUM_AGGS:
        model.album_agg = agg
        album = per_customer_metrics(get_rankings(model, custs, "album", K), rel["album"], [K]).mean()
        rows.append(dict(
            neg_ratio=params["neg_ratio"], n_folds=params["n_folds"], max_depth=params["max_depth"],
            learning_rate=gbdt["learning_rate"], max_iter=gbdt["max_iter"], album_agg="+".join(agg),
            seed=seed,
            album_ndcg=album[f"ndcg@{K}"], album_recall=album[f"recall@{K}"], album_hitrate=album[f"hitrate@{K}"],
            track_ndcg=track[f"ndcg@{K}"], track_recall=track[f"recall@{K}"], track_hitrate=track[f"hitrate@{K}"],
        ))
    return rows


def main():
    split, customers = split_by_time(), CustomerDAO.get_all()
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    jobs = [(p, s) for p in combos for s in TUNE_SEEDS]
    out = Parallel(n_jobs=-1, verbose=5)(delayed(run_one)(p, s, split, customers) for p, s in jobs)
    raw = pd.DataFrame([r for rows in out for r in rows])

    keys = ["neg_ratio", "n_folds", "max_depth", "learning_rate", "max_iter", "album_agg"]
    metrics = [c for c in raw.columns if c not in keys + ["seed"]]
    agg = raw.groupby(keys)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{'mean' if s == 'mean' else 'std_over_seeds'}" for m, s in agg.columns]
    res = agg.reset_index().sort_values(["album_ndcg_mean", "album_recall_mean"], ascending=False)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    res.to_csv(ARTIFACTS_DIR / "tuning.csv", index=False)

    pd.set_option("display.width", 220)
    cols = keys + ["album_ndcg_mean", "album_ndcg_std_over_seeds", "album_recall_mean", "track_hitrate_mean"]
    print(f"{len(res)} cấu hình x {len(TUNE_SEEDS)} seed. Top 10 theo album NDCG@{K} (val):")
    print(res[cols].head(10).round(4).to_string(index=False))
    print(f"\nTrung vị toàn lưới: album NDCG@{K} = {res.album_ndcg_mean.median():.4f}")


if __name__ == "__main__":
    main()
