"""Trạng thái fit trên train và các feature cho cặp (customer, track).

Mọi thống kê trong `State` chỉ được fit trên dữ liệu `train_like` truyền vào
(train, hoặc train+val ở lần huấn luyện cuối) — không bao giờ nhìn vào test.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Toàn bộ feature của Hybrid Ranker (TECHSPEC mục 5)
FEATURE_COLUMNS = [
    "pop_track", "pop_album", "pop_artist",
    "genre_affinity", "artist_affinity", "album_affinity",
    "content_cosine", "knn_score", "mf_score",
    "country_genre_pop", "unit_price", "milliseconds_log",
    "same_media_type_ratio", "days_since_last_purchase",
]
# Feature tính được ở đây; content_cosine / knn_score / mf_score do B3-B5 cung cấp
MODEL_FEATURES = ["content_cosine", "knn_score", "mf_score"]
BASE_FEATURES = [c for c in FEATURE_COLUMNS if c not in MODEL_FEATURES]


@dataclass
class State:
    tracks: pd.DataFrame            # thuộc tính tĩnh, index = track_id
    hist: pd.DataFrame              # lịch sử train_like, đã gắn thuộc tính track
    hist_by_customer: dict          # customer_id -> DataFrame con của hist
    pop_track: pd.Series            # số khách mua, index = track_id (đủ mọi track)
    pop_album: pd.Series            # số khách mua >=1 track của album
    pop_artist: pd.Series
    country_of: dict                # customer_id -> country
    country_genre: pd.DataFrame     # hàng = country, cột = genre_id, giá trị = tỷ lệ lượt mua
    country_track_pop: dict         # country -> Series số khách mua (cho cold-start)
    last_purchase: pd.Series        # customer_id -> ngày mua cuối
    ref_date: pd.Timestamp          # mốc tính days_since_last_purchase

    def has_history(self, customer_id):
        return customer_id in self.hist_by_customer

    def bought_tracks(self, customer_id):
        h = self.hist_by_customer.get(customer_id)
        return set() if h is None else set(h.track_id)


def fit_state(train_like, tracks, customers):
    """Fit mọi bảng thống kê trên `train_like`. `customers` là list[Customer]."""
    attrs = ["track_id", "album_id", "artist_id", "genre_id", "media_type_id"]
    hist = train_like.merge(tracks[attrs], on="track_id", how="left")

    def distinct_customers(col):
        return hist.drop_duplicates(["customer_id", col]).groupby(col).size()

    pop_track = distinct_customers("track_id").reindex(tracks.index, fill_value=0)
    pop_album = distinct_customers("album_id")
    pop_artist = distinct_customers("artist_id")

    country_of = {c.customer_id: c.country for c in customers}
    hist["country"] = hist.customer_id.map(country_of)
    cg = hist.groupby(["country", "genre_id"]).size().unstack(fill_value=0)
    country_genre = cg.div(cg.sum(axis=1), axis=0)
    country_track_pop = {
        country: g.drop_duplicates(["customer_id", "track_id"]).groupby("track_id").size()
        for country, g in hist.groupby("country")
    }

    return State(
        tracks=tracks,
        hist=hist,
        hist_by_customer={c: g for c, g in hist.groupby("customer_id")},
        pop_track=pop_track,
        pop_album=pop_album,
        pop_artist=pop_artist,
        country_of=country_of,
        country_genre=country_genre,
        country_track_pop=country_track_pop,
        last_purchase=hist.groupby("customer_id").invoice_date.max(),
        ref_date=hist.invoice_date.max(),
    )


def base_features(state, customer_id, candidate_ids):
    """Feature của khách với từng track ứng viên (không gồm 3 feature từ mô hình B3-B5).

    Trả về DataFrame index = candidate_ids, cột = BASE_FEATURES.
    """
    cand = state.tracks.loc[candidate_ids]
    h = state.hist_by_customer.get(customer_id)
    n = 0 if h is None else len(h)

    def share(col):
        """Tỷ lệ lượt mua của khách rơi vào cùng giá trị `col` với track ứng viên."""
        if n == 0:
            return np.zeros(len(cand))
        return cand[col].map(h[col].value_counts() / n).fillna(0.0).to_numpy()

    country = state.country_of.get(customer_id)
    if country in state.country_genre.index:
        cgp = cand.genre_id.map(state.country_genre.loc[country]).fillna(0.0).to_numpy()
    else:
        cgp = np.zeros(len(cand))

    if customer_id in state.last_purchase.index:
        days = (state.ref_date - state.last_purchase[customer_id]).days
    else:
        days = np.nan

    return pd.DataFrame({
        "pop_track": state.pop_track.reindex(cand.index).to_numpy(),
        "pop_album": cand.album_id.map(state.pop_album).fillna(0).to_numpy(),
        "pop_artist": cand.artist_id.map(state.pop_artist).fillna(0).to_numpy(),
        "genre_affinity": share("genre_id"),
        "artist_affinity": share("artist_id"),
        "album_affinity": share("album_id"),
        "country_genre_pop": cgp,
        "unit_price": cand.unit_price.to_numpy(),
        "milliseconds_log": np.log1p(cand.milliseconds.to_numpy()),
        "same_media_type_ratio": share("media_type_id"),
        "days_since_last_purchase": days,
    }, index=cand.index)[BASE_FEATURES]
