"""Đánh giá: metric @K theo từng khách, bootstrap CI, paired bootstrap (TECHSPEC mục 6).

Quy ước chia dữ liệu khi đánh giá:
  - which="val"  : fit trên train, đo trên val (chọn siêu tham số);
  - which="test" : fit trên train+val, đo trên test (kết quả cuối).
Nếu fit trên train mà đo trên test thì val sẽ lọt vào candidate set một cách vô lý.
"""
import warnings

import numpy as np
import pandas as pd

from config import BOOTSTRAP_ROUNDS, CI_LEVEL, SEED, TOP_K_LIST
from ml.data_prep import add_album_interactions

METRICS = ["recall", "hitrate", "ndcg", "map"]
LEVELS = ["track", "album"]
GROUPS = ["all", "seen", "cold"]  # seen/cold: item có/không có người mua trong dữ liệu fit


# ---------------------------------------------------------------- metric từng khách
def _metrics_at_k(ranked, relevant, k):
    """Recall, HitRate, NDCG, MAP @k cho một khách. `relevant` khác rỗng."""
    top = ranked[:k]
    hit = np.array([i in relevant for i in top], dtype=float)
    n_rel = len(relevant)
    denom = min(n_rel, k)
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float((hit * discounts).sum())
    idcg = float((1.0 / np.log2(np.arange(2, denom + 2))).sum())
    precision_at_i = np.cumsum(hit) / np.arange(1, len(top) + 1)
    return {
        "recall": hit.sum() / n_rel,
        "hitrate": float(hit.any()),
        "ndcg": dcg / idcg,
        "map": float((precision_at_i * hit).sum() / denom),
    }


def per_customer_metrics(rankings, relevant, ks=TOP_K_LIST):
    """DataFrame index = customer, cột = 'metric@K'. Khách không có item liên quan -> NaN."""
    rows = {}
    for c, rel in relevant.items():
        if not rel:
            rows[c] = {f"{m}@{k}": np.nan for k in ks for m in METRICS}
            continue
        row = {}
        for k in ks:
            for m, v in _metrics_at_k(rankings[c], rel, k).items():
                row[f"{m}@{k}"] = v
        rows[c] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


# ---------------------------------------------------------------- tập item liên quan
def relevant_sets(target, tracks, fit_data, level="track", group="all"):
    """customer -> set item thật sự mua ở `target`, lọc theo nhóm seen/cold.

    seen = item đã có người mua trong `fit_data`; cold = chưa ai mua.
    Mọi khách trong `fit_data` đều có mặt (tập rỗng nếu không còn item nào).
    """
    if level == "track":
        tgt = target[["customer_id", "track_id"]].rename(columns={"track_id": "item"})
        seen_items = set(fit_data.track_id)
    else:
        tgt = add_album_interactions(target, tracks).rename(columns={"album_id": "item"})
        seen_items = set(add_album_interactions(fit_data, tracks).album_id)

    if group == "seen":
        tgt = tgt[tgt["item"].isin(seen_items)]
    elif group == "cold":
        tgt = tgt[~tgt["item"].isin(seen_items)]

    rel = {c: set() for c in sorted(set(fit_data.customer_id))}
    for c, g in tgt.groupby("customer_id"):
        rel[c] = set(g["item"])
    return rel


# ---------------------------------------------------------------- xếp hạng & diagnostic
def get_rankings(model, customers, level="track", k_max=max(TOP_K_LIST)):
    recommend = model.recommend_tracks if level == "track" else model.recommend_albums
    return {c: [r.item_id for r in recommend(c, k_max)] for c in customers}


def diagnostic_rows(rankings, state, level, ks=TOP_K_LIST):
    """Coverage và Novelty (mức toàn tập, chỉ để chẩn đoán).

    Coverage@K = tỷ lệ item trong catalog xuất hiện ít nhất một lần ở top-K của mọi khách.
    Novelty@K  = trung bình -log2 của độ phổ biến (làm trơn) của item được gợi ý.
    """
    n_cust = len(state.hist_by_customer)
    if level == "track":
        pop, catalog = state.pop_track, len(state.tracks)
    else:
        pop, catalog = state.pop_album, state.tracks.album_id.nunique()
    rows = []
    for k in ks:
        items = [i for r in rankings.values() for i in r[:k]]
        p = (pop.reindex(items).fillna(0).to_numpy() + 1) / (n_cust + 1)
        rows.append(("coverage", k, len(set(items)) / catalog))
        rows.append(("novelty", k, float(-np.log2(p).mean())))
    return rows


# ---------------------------------------------------------------- bootstrap
def _bootstrap_idx(n, rounds, seed):
    return np.random.default_rng(seed).integers(0, n, size=(rounds, n))


def _boot_means(values, idx):
    """values: (n,) hoặc (n, m). Trả về trung bình mỗi lần resample, bỏ qua NaN."""
    # resample có thể toàn NaN (nhóm có ít khách) -> nanmean cho NaN, không cần cảnh báo
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values[idx], axis=1)


def _ci(boot, level=CI_LEVEL):
    a = (1 - level) / 2
    return np.nanquantile(boot, a, axis=0), np.nanquantile(boot, 1 - a, axis=0)


def summarize(per_cust, rounds=BOOTSTRAP_ROUNDS, seed=SEED):
    """Bảng mean + CI bootstrap (resample khách) cho từng cột metric@K."""
    vals = per_cust.to_numpy(dtype=float)
    idx = _bootstrap_idx(len(vals), rounds, seed)
    lo, hi = _ci(_boot_means(vals, idx))
    out = pd.DataFrame({
        "mean": np.nanmean(vals, axis=0), "ci_lo": lo, "ci_hi": hi,
        "n_customers": (~np.isnan(vals)).sum(axis=0),
    }, index=per_cust.columns)
    out.index.name = "metric_k"
    return out


def paired_bootstrap(per_cust_main, per_cust_base, rounds=BOOTSTRAP_ROUNDS, seed=SEED):
    """M - baseline: hiệu trung bình, CI và p-value hai phía (cùng một lần resample khách)."""
    diff = per_cust_main.to_numpy(dtype=float) - per_cust_base.to_numpy(dtype=float)
    idx = _bootstrap_idx(len(diff), rounds, seed)
    boot = _boot_means(diff, idx)
    lo, hi = _ci(boot)
    # +1 để p-value không bao giờ bằng 0 với số lần resample hữu hạn
    p_le = ((boot <= 0).sum(axis=0) + 1) / (rounds + 1)
    p_ge = ((boot >= 0).sum(axis=0) + 1) / (rounds + 1)
    out = pd.DataFrame({
        "diff": np.nanmean(diff, axis=0), "ci_lo": lo, "ci_hi": hi,
        "p_value": np.minimum(1.0, 2 * np.minimum(p_le, p_ge)),
    }, index=per_cust_main.columns)
    out.index.name = "metric_k"
    return out


def holm(p_values):
    """Hiệu chỉnh Holm-Bonferroni; trả về mảng p đã hiệu chỉnh theo thứ tự đầu vào."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------- điều phối
def evaluate_models(models, split, which="val", main=None, levels=LEVELS, groups=GROUPS,
                    ks=TOP_K_LIST, rounds=BOOTSTRAP_ROUNDS, seed=SEED, customers=None):
    """Fit từng mô hình, đánh giá, trả về (metrics_df, paired_df).

    `models`: list Recommender chưa fit. `main`: tên mô hình chính để so cặp với các mô hình còn lại
    (None -> không tính paired). Holm áp dụng trong từng (level, group) trên mọi dòng so sánh.
    """
    if which == "val":
        fit_data, target = split.train, split.val
    elif which == "test":
        fit_data, target = split.train_plus_val(), split.test
    else:
        raise ValueError("which phải là 'val' hoặc 'test'")

    customers = customers if customers is not None else []
    all_customers = sorted(set(fit_data.customer_id))
    metric_rows, per_cust = [], {}  # per_cust[(level, group)][model_name] -> DataFrame

    for model in models:
        model.fit(fit_data, split.tracks, customers)
        for level in levels:
            rankings = get_rankings(model, all_customers, level, max(ks))
            for metric, k, value in diagnostic_rows(rankings, model.state, level, ks):
                metric_rows.append(dict(model=model.name, level=level, group="all", metric=metric,
                                        k=k, mean=value, ci_lo=np.nan, ci_hi=np.nan, n_customers=np.nan))
            for group in groups:
                rel = relevant_sets(target, split.tracks, fit_data, level, group)
                pc = per_customer_metrics(rankings, rel, ks)
                per_cust.setdefault((level, group), {})[model.name] = pc
                summ = summarize(pc, rounds, seed).reset_index()
                summ[["metric", "k"]] = summ.metric_k.str.split("@", expand=True)
                summ["k"] = summ["k"].astype(int)
                for r in summ.itertuples():
                    metric_rows.append(dict(model=model.name, level=level, group=group, metric=r.metric,
                                            k=r.k, mean=r.mean, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                                            n_customers=r.n_customers))
    metrics_df = pd.DataFrame(metric_rows)

    paired_rows = []
    if main is not None:
        for (level, group), by_model in per_cust.items():
            block = []
            for name, pc in by_model.items():
                if name == main:
                    continue
                res = paired_bootstrap(by_model[main], pc, rounds, seed).reset_index()
                res[["metric", "k"]] = res.metric_k.str.split("@", expand=True)
                res["k"] = res["k"].astype(int)
                res.insert(0, "baseline", name)
                res.insert(0, "main", main)
                res.insert(2, "level", level)
                res.insert(3, "group", group)
                block.append(res.drop(columns="metric_k"))
            if block:
                block = pd.concat(block, ignore_index=True)
                block["p_holm"] = holm(block.p_value.fillna(1.0))
                paired_rows.append(block)
    paired_df = pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    return metrics_df, paired_df


def aggregate_seeds(frames):
    """frames: {seed: metrics_df}. Trả về mean ± std của cột `mean` qua các seed."""
    long = pd.concat([f.assign(seed=s) for s, f in frames.items()], ignore_index=True)
    keys = ["model", "level", "group", "metric", "k"]
    g = long.groupby(keys)["mean"].agg(["mean", "std"]).reset_index()
    return g.rename(columns={"mean": "mean_over_seeds", "std": "std_over_seeds"})


if __name__ == "__main__":
    from dal.CustomerDAO import CustomerDAO
    from ml.data_prep import split_by_time
    from ml.hybrid import HybridRecommender
    from ml.recommenders import (ALSRecommender, ContentBasedRecommender, GenreArtistPopRecommender,
                                 ItemKNNRecommender, PopularityRecommender, RandomRecommender)

    split = split_by_time()
    models = [RandomRecommender(), PopularityRecommender(), GenreArtistPopRecommender(),
              ItemKNNRecommender(), ContentBasedRecommender(), ALSRecommender(),
              HybridRecommender()]
    m, p = evaluate_models(models, split, which="val", main="M_hybrid",
                           customers=CustomerDAO.get_all())
    pd.set_option("display.width", 200)
    sel = m[(m.group == "all") & (m.k == 10) & m.metric.isin(METRICS)]
    print(sel.pivot_table(index=["level", "model"], columns="metric", values="mean").round(4))
    print(p[(p.group == "all") & (p.k == 10) & (p.metric == "recall")].round(4).to_string(index=False))
