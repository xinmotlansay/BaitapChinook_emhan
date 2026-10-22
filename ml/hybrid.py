"""Mô hình chính M: Hybrid Ranker (pointwise GBDT) trên feature kết hợp tín hiệu của B1-B5.

Chống rò rỉ khi dựng tập huấn luyện của ranker: với mỗi "fold trong", feature của
(khách, track) được tính từ các mô hình/thống kê fit trên hóa đơn *trước* hóa đơn đích;
nhãn dương là track mua ở hóa đơn đích. Mô hình cuối cùng mới được refit trên toàn bộ dữ liệu fit.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from config import ALBUM_AGG, GBDT_PARAMS, HYBRID_FOLDS, NEG_RATIO, SEED
from ml.features import FEATURE_COLUMNS, base_features
from ml.recommenders import (ALSRecommender, ContentBasedRecommender, ItemKNNRecommender,
                             Recommender)


class _Components:
    """Bộ B3/B4/B5 + trạng thái chung, cùng fit trên một tập dữ liệu."""

    def __init__(self, fit_data, tracks, customers, seed):
        self.knn = ItemKNNRecommender(seed).fit(fit_data, tracks, customers)
        self.content = ContentBasedRecommender(seed).fit(fit_data, tracks, customers)
        self.als = ALSRecommender(seed).fit(fit_data, tracks, customers)
        self.state = self.knn.state  # các mô hình fit trên cùng dữ liệu nên state giống nhau

    def features(self, customer_id, ids):
        """DataFrame (index = ids) theo đúng thứ tự FEATURE_COLUMNS."""
        f = base_features(self.state, customer_id, ids)
        f["content_cosine"] = self.content._score(customer_id, ids)
        f["knn_score"] = self.knn._score(customer_id, ids)
        f["mf_score"] = self.als._score(customer_id, ids)
        return f[FEATURE_COLUMNS]


def inner_folds(fit_data, n_folds):
    """Sinh (fit, target) theo leave-last-invoice-out: fold j lấy hóa đơn thứ j tính từ cuối làm đích."""
    inv = (fit_data[["customer_id", "invoice_id", "invoice_date"]].drop_duplicates("invoice_id")
           .sort_values(["customer_id", "invoice_date", "invoice_id"]))
    rank = inv.assign(r=inv.groupby("customer_id").cumcount(ascending=False)).set_index("invoice_id")["r"]
    r = fit_data.invoice_id.map(rank)
    for j in range(n_folds):
        fit, target = fit_data[r > j], fit_data[r == j]
        # chỉ giữ khách còn lịch sử để fit
        target = target[target.customer_id.isin(set(fit.customer_id))]
        # assert chống rò rỉ: mỗi khách, hóa đơn fit đều trước hóa đơn đích
        last_fit = fit.groupby("customer_id").invoice_date.max()
        first_tgt = target.groupby("customer_id").invoice_date.min()
        assert (last_fit.loc[first_tgt.index] < first_tgt).all(), "Rò rỉ ở fold trong của Hybrid"
        yield fit, target


class HybridRecommender(Recommender):
    name = "M_hybrid"

    def __init__(self, seed=SEED, neg_ratio=NEG_RATIO, n_folds=HYBRID_FOLDS, album_agg=ALBUM_AGG,
                 gbdt_params=None):
        super().__init__(seed)
        self.neg_ratio, self.n_folds, self.album_agg = neg_ratio, n_folds, tuple(album_agg)
        self.gbdt_params = dict(GBDT_PARAMS if gbdt_params is None else gbdt_params)
        self.clf = None
        self.train_info = {}

    # ---- huấn luyện ----
    def fit(self, train_like, tracks, customers):
        super().fit(train_like, tracks, customers)  # state cho candidate set / cold-start
        rng = np.random.default_rng(self.seed)
        xs, ys = [], []
        for fit, target in inner_folds(train_like, self.n_folds):
            comp = _Components(fit, tracks, customers, self.seed)
            for c, g in target.groupby("customer_id"):
                pos = sorted(set(g.track_id))
                cand = comp.knn.candidates(c).difference(pos)
                neg = rng.choice(cand, size=min(len(cand), self.neg_ratio * len(pos)), replace=False)
                ids = pd.Index(pos).append(pd.Index(neg))
                xs.append(comp.features(c, ids))
                ys.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
        x, y = pd.concat(xs), np.concatenate(ys)
        self.clf = HistGradientBoostingClassifier(random_state=self.seed, **self.gbdt_params).fit(x, y)
        self.train_info = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "n_folds": self.n_folds}
        self.comp = _Components(train_like, tracks, customers, self.seed)  # refit trên toàn bộ dữ liệu fit
        return self

    # ---- dự đoán ----
    def _score(self, customer_id, candidate_ids):
        return self.clf.predict_proba(self.comp.features(customer_id, candidate_ids))[:, 1]

    def _reasons(self, customer_id, track_ids):
        """Lý do theo thứ tự ưu tiên tín hiệu: cùng nghệ sĩ > đồng mua (kNN) > MF > nội dung > phổ biến."""
        ids = pd.Index(track_ids)
        f = self.comp.features(customer_id, ids)
        t = self.state.tracks
        knn = self.comp.knn._reasons(customer_id, ids)
        content = self.comp.content._reasons(customer_id, ids)
        out = []
        for i, tid in enumerate(ids):
            row = f.loc[tid]
            if row.artist_affinity > 0:
                out.append(f"Cùng nghệ sĩ {t.at[tid, 'artist_name']} với bạn đã mua")
            elif row.knn_score > 0:
                out.append(knn[i])
            elif row.mf_score > 0.05:
                out.append("Khách có hành vi mua tương tự bạn cũng mua")
            elif row.content_cosine > 0:
                out.append(content[i])
            else:
                out.append("Phổ biến với các khách khác")
        return out

    def _album_rank_scores(self, cand, track_scores, best):
        """Điểm album = tổng các phép gộp trong `album_agg` (mặc định max + mean) trên track của album."""
        album_ids = self.state.tracks.loc[cand, "album_id"].to_numpy()
        df = pd.DataFrame({"a": album_ids, "s": track_scores})
        g = df.groupby("a")["s"]
        total = sum(g.agg(name) for name in self.album_agg)
        return total.to_dict()
