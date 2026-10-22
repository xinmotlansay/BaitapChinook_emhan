"""Các mô hình gợi ý: baseline B0-B5 (Hybrid Ranker nằm ở hybrid.py)."""
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ALS_ALPHA, ALS_FACTORS, ALS_ITERS, ALS_REG, DEFAULT_TOP_K, SEED
from ml.features import base_features, fit_state
from models.entities import Recommendation


class Recommender:
    """Lớp cha: lo candidate set, cold-start khách, và gộp track -> album.

    Lớp con chỉ cần cài `_score` (và `_reasons` nếu muốn lý do riêng).
    """
    name = "base"

    def __init__(self, seed=SEED):
        self.seed = seed
        self.state = None

    def fit(self, train_like, tracks, customers):
        self.state = fit_state(train_like, tracks, customers)
        return self

    # ---- điểm số ----
    def _score(self, customer_id, candidate_ids):
        raise NotImplementedError

    def _reasons(self, customer_id, track_ids):
        return ["Phổ biến với các khách khác"] * len(track_ids)

    def score_tracks(self, customer_id, candidate_ids):
        """Điểm cho từng ứng viên; khách chưa có lịch sử -> fallback phổ biến theo quốc gia."""
        if not self.state.has_history(customer_id):
            return self._cold_scores(customer_id, candidate_ids)
        return np.asarray(self._score(customer_id, candidate_ids), dtype=float)

    def _cold_scores(self, customer_id, candidate_ids):
        s = self.state
        pop = s.country_track_pop.get(s.country_of.get(customer_id))
        local = np.zeros(len(candidate_ids)) if pop is None else pop.reindex(candidate_ids, fill_value=0).to_numpy()
        # độ phổ biến toàn cục chỉ dùng để phá hòa
        return local + 1e-3 * s.pop_track.reindex(candidate_ids).to_numpy()

    # ---- xếp hạng ----
    def candidates(self, customer_id):
        """Mọi track khách chưa mua trong dữ liệu fit (không gợi ý lại)."""
        return self.state.tracks.index.difference(list(self.state.bought_tracks(customer_id)))

    def recommend_tracks(self, customer_id, k=DEFAULT_TOP_K):
        cand = self.candidates(customer_id)
        scores = self.score_tracks(customer_id, cand)
        top = np.argsort(-scores, kind="stable")[:k]
        ids = cand[top]
        reasons = self._reasons(customer_id, ids) if self.state.has_history(customer_id) \
            else ["Phổ biến tại quốc gia của bạn"] * len(ids)
        t = self.state.tracks
        return [
            Recommendation(int(i), t.at[i, "name"], t.at[i, "artist_name"], float(scores[j]), r)
            for i, j, r in zip(ids, top, reasons)
        ]

    def _album_scores(self, customer_id, cand, track_scores):
        """Mặc định: điểm album = max điểm các track của album."""
        album_ids = self.state.tracks.loc[cand, "album_id"].to_numpy()
        best = {}
        for pos, (a, s) in enumerate(zip(album_ids, track_scores)):
            if a not in best or s > track_scores[best[a]]:
                best[a] = pos
        return best  # album_id -> vị trí track tốt nhất trong cand

    def recommend_albums(self, customer_id, k=DEFAULT_TOP_K):
        t = self.state.tracks
        bought_albums = set(t.loc[list(self.state.bought_tracks(customer_id)), "album_id"])
        cand = self.candidates(customer_id)
        scores = self.score_tracks(customer_id, cand)
        best = self._album_scores(customer_id, cand, scores)
        album_score = self._album_rank_scores(cand, scores, best)
        albums = [a for a in best if a not in bought_albums]
        albums.sort(key=lambda a: -album_score[a])  # sort ổn định
        out = []
        for a in albums[:k]:
            pos = best[a]
            tid = cand[pos]
            reason = self._reasons(customer_id, [tid])[0] \
                if self.state.has_history(customer_id) else "Phổ biến tại quốc gia của bạn"
            out.append(Recommendation(int(a), t.at[tid, "album_title"], t.at[tid, "artist_name"],
                                      float(album_score[a]), reason))
        return out

    def _album_rank_scores(self, cand, track_scores, best):
        """Điểm dùng để xếp hạng album. Mặc định = điểm track tốt nhất; lớp con có thể gộp khác."""
        return {a: track_scores[pos] for a, pos in best.items()}


class RandomRecommender(Recommender):
    """B0: sàn so sánh. Xác định theo (seed, customer) để tái lập."""
    name = "B0_random"

    def _rng(self, customer_id):
        return np.random.default_rng([self.seed, int(customer_id)])

    def _score(self, customer_id, candidate_ids):
        return self._rng(customer_id).random(len(candidate_ids))

    def _cold_scores(self, customer_id, candidate_ids):
        return self._score(customer_id, candidate_ids)

    def _reasons(self, customer_id, track_ids):
        return ["Ngẫu nhiên (baseline)"] * len(track_ids)

    def _album_scores(self, customer_id, cand, track_scores):
        # chọn album đồng đều, không thiên vị album nhiều track
        album_ids = self.state.tracks.loc[cand, "album_id"].to_numpy()
        rng = self._rng(customer_id)
        uniq = np.unique(album_ids)
        album_rand = dict(zip(uniq, rng.random(len(uniq))))
        track_scores[:] = [album_rand[a] for a in album_ids]
        return super()._album_scores(customer_id, cand, track_scores)


class PopularityRecommender(Recommender):
    """B1: số khách đã mua track trong train."""
    name = "B1_popularity"

    def _score(self, customer_id, candidate_ids):
        return self.state.pop_track.reindex(candidate_ids).to_numpy()

    def _reasons(self, customer_id, track_ids):
        pops = self.state.pop_track.reindex(track_ids)
        return [f"{int(p)} khách khác đã mua" for p in pops]


class GenreArtistPopRecommender(Recommender):
    """B2: (độ ưa thích thể loại + nghệ sĩ của khách) x độ phổ biến.

    Độ phổ biến dùng `1 + log1p(pop_track)` để track chưa ai mua (item-cold)
    vẫn nhận điểm theo sở thích của khách.
    """
    name = "B2_genre_artist_pop"

    def _score(self, customer_id, candidate_ids):
        f = base_features(self.state, customer_id, candidate_ids)
        return ((f.genre_affinity + f.artist_affinity) * (1 + np.log1p(f.pop_track))).to_numpy()

    def _reasons(self, customer_id, track_ids):
        f = base_features(self.state, customer_id, track_ids)
        t = self.state.tracks
        out = []
        for tid, row in f.iterrows():
            if row.artist_affinity > 0:
                out.append(f"Cùng nghệ sĩ {t.at[tid, 'artist_name']} với bạn đã mua")
            elif row.genre_affinity > 0:
                out.append(f"Thể loại {t.at[tid, 'genre_name']} chiếm {row.genre_affinity:.0%} lượt mua của bạn")
            else:
                out.append("Phổ biến với các khách khác")
        return out


class ItemKNNRecommender(Recommender):
    """B3: Item-kNN cosine trên đồng xuất hiện mức album (đỡ thưa hơn mức track).

    Điểm album a của khách = tổng độ tương đồng cosine(a, b) với các album b khách đã mua;
    mọi track của album thừa hưởng điểm của album.
    """
    name = "B3_item_knn"

    def fit(self, train_like, tracks, customers):
        super().fit(train_like, tracks, customers)
        h = self.state.hist
        self.album_ids = np.sort(tracks.album_id.unique())
        self.album_pos = {a: i for i, a in enumerate(self.album_ids)}
        cust = {c: i for i, c in enumerate(sorted(h.customer_id.unique()))}
        pairs = h.drop_duplicates(["customer_id", "album_id"])
        rows = pairs.customer_id.map(cust).to_numpy()
        cols = pairs.album_id.map(self.album_pos).to_numpy()
        x = sparse.csr_matrix((np.ones(len(pairs)), (rows, cols)), shape=(len(cust), len(self.album_ids)))
        co = (x.T @ x).toarray()
        norm = np.sqrt(np.diag(co))
        norm[norm == 0] = 1.0
        self.sim = co / np.outer(norm, norm)
        np.fill_diagonal(self.sim, 0.0)  # album không tự gợi ý bằng chính nó
        return self

    def _user_vec(self, customer_id):
        u = np.zeros(len(self.album_ids))
        for a in self.state.hist_by_customer[customer_id].album_id.unique():
            u[self.album_pos[a]] = 1.0
        return u

    def _score(self, customer_id, candidate_ids):
        album_score = self.sim @ self._user_vec(customer_id)
        pos = self.state.tracks.loc[candidate_ids, "album_id"].map(self.album_pos).to_numpy()
        return album_score[pos]

    def _reasons(self, customer_id, track_ids):
        u = self._user_vec(customer_id)
        t = self.state.tracks
        title = t.drop_duplicates("album_id").set_index("album_id")["album_title"]
        out = []
        for tid in track_ids:
            a = self.album_pos[t.at[tid, "album_id"]]
            contrib = self.sim[a] * u
            if contrib.max() <= 0:
                out.append("Phổ biến với các khách khác")
            else:
                out.append(f"Khách mua album «{title[self.album_ids[contrib.argmax()]]}» cũng hay mua album này")
        return out


def _identity(doc):
    return doc  # analyzer cho TfidfVectorizer; định nghĩa ở mức module để pickle được


def _content_tokens(row):
    """Token nội dung của một track: thể loại, nghệ sĩ, định dạng, nhà soạn nhạc."""
    toks = [f"g:{row.genre_id}", f"a:{row.artist_id}", f"m:{row.media_type_id}"]
    for c in re.split(r"\s*[,/;&]\s*", row.composer or ""):
        c = c.strip().lower()
        if c:
            toks.append(f"c:{c}")
    return toks


_TOKEN_LABEL = {"g": "thể loại", "a": "nghệ sĩ", "m": "định dạng", "c": "nhà soạn nhạc"}


class ContentBasedRecommender(Recommender):
    """B4: TF-IDF (thể loại, nghệ sĩ, định dạng, nhà soạn nhạc) -> hồ sơ khách = trung bình -> cosine.

    idf chỉ fit trên các track được mua trong dữ liệu train; token chưa từng xuất hiện
    ở đó bị bỏ (hồ sơ khách cũng không chứa chúng nên không mất tín hiệu).
    Hoạt động cho cả item-cold vì chỉ dựa vào thuộc tính tĩnh.
    """
    name = "B4_content"

    def fit(self, train_like, tracks, customers):
        super().fit(train_like, tracks, customers)
        docs = [_content_tokens(r) for r in tracks.itertuples()]
        self.row_of = {tid: i for i, tid in enumerate(tracks.index)}
        bought_rows = [self.row_of[t] for t in sorted(set(self.state.hist.track_id))]
        self.vec = TfidfVectorizer(analyzer=_identity)
        self.vec.fit([docs[i] for i in bought_rows])
        self.x = self.vec.transform(docs).tocsr()  # hàng đã chuẩn hóa L2
        self.vocab = np.array(self.vec.get_feature_names_out())
        return self

    def _profile(self, customer_id):
        rows = [self.row_of[t] for t in self.state.hist_by_customer[customer_id].track_id]
        p = np.asarray(self.x[rows].mean(axis=0)).ravel()
        n = np.linalg.norm(p)
        return p / n if n > 0 else p

    def _score(self, customer_id, candidate_ids):
        rows = [self.row_of[t] for t in candidate_ids]
        return self.x[rows] @ self._profile(customer_id)

    def _reasons(self, customer_id, track_ids):
        p = self._profile(customer_id)
        t = self.state.tracks
        out = []
        for tid in track_ids:
            contrib = self.x[self.row_of[tid]].multiply(p).toarray().ravel()
            if contrib.max() <= 0:
                out.append("Phổ biến với các khách khác")
                continue
            kind, val = self.vocab[contrib.argmax()].split(":", 1)
            if kind == "m":  # Track chưa có tên định dạng, chỉ có mã
                out.append("Giống nội dung bạn đã nghe: cùng định dạng file")
                continue
            shown = {"g": t.at[tid, "genre_name"], "a": t.at[tid, "artist_name"], "c": val}[kind]
            out.append(f"Giống nội dung bạn đã nghe: {_TOKEN_LABEL[kind]} {shown}")
        return out


class ALSRecommender(Recommender):
    """B5: ALS cho implicit feedback (Hu, Koren, Volinsky 2008) tự viết bằng numpy, mức track.

    Track chưa ai mua trong dữ liệu fit có vector nhân tố bằng 0 nên điểm = 0
    (MF không xếp hạng được item-cold) — đây là lý do cần Hybrid.
    """
    name = "B5_als"

    def __init__(self, seed=SEED, factors=ALS_FACTORS, reg=ALS_REG, iters=ALS_ITERS, alpha=ALS_ALPHA):
        super().__init__(seed)
        self.factors, self.reg, self.iters, self.alpha = factors, reg, iters, alpha

    def fit(self, train_like, tracks, customers):
        super().fit(train_like, tracks, customers)
        h = self.state.hist
        self.cust_row = {c: i for i, c in enumerate(sorted(h.customer_id.unique()))}
        self.item_row = {t: i for i, t in enumerate(tracks.index)}
        r = np.zeros((len(self.cust_row), len(self.item_row)))
        r[h.customer_id.map(self.cust_row).to_numpy(), h.track_id.map(self.item_row).to_numpy()] = 1.0

        rng = np.random.default_rng(self.seed)
        self.user_f = rng.normal(0, 0.01, (r.shape[0], self.factors))
        self.item_f = rng.normal(0, 0.01, (r.shape[1], self.factors))
        for _ in range(self.iters):
            self.user_f = self._solve(r, self.item_f)
            self.item_f = self._solve(r.T, self.user_f)
        return self

    def _solve(self, r, other):
        """Cập nhật một phía: với mỗi hàng u giải (YtY + a*Yu'Yu + reg*I) x = (1+a)*sum(Yu)."""
        yty = other.T @ other + self.reg * np.eye(self.factors)
        out = np.zeros((r.shape[0], self.factors))
        for u in range(r.shape[0]):
            idx = np.flatnonzero(r[u])
            if len(idx) == 0:
                continue  # không có tương tác -> vector 0
            y = other[idx]
            a = yty + self.alpha * (y.T @ y)
            out[u] = np.linalg.solve(a, (1 + self.alpha) * y.sum(axis=0))
        return out

    def _score(self, customer_id, candidate_ids):
        rows = [self.item_row[t] for t in candidate_ids]
        return self.item_f[rows] @ self.user_f[self.cust_row[customer_id]]

    def _reasons(self, customer_id, track_ids):
        return ["Khách có hành vi mua tương tự bạn cũng mua"] * len(track_ids)
