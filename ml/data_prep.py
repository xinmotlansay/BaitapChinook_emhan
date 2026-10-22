"""Chuẩn bị dữ liệu: chia train/val/test theo thời gian và kiểm tra rò rỉ (TECHSPEC mục 4)."""
from dataclasses import dataclass

import pandas as pd

from dal.InteractionDAO import InteractionDAO
from dal.TrackDAO import TrackDAO

MIN_INVOICES = 3  # cần ít nhất train + val + test


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    tracks: pd.DataFrame  # thuộc tính tĩnh của track (kèm album_id, artist_id, genre_id...)

    def train_plus_val(self):
        """Dữ liệu huấn luyện lần cuối (train + val), đánh giá trên test."""
        return pd.concat([self.train, self.val], ignore_index=True)


def load_tracks():
    """Bảng thuộc tính track. Không phụ thuộc thời gian nên dùng cho mọi tập."""
    rows = [t.__dict__ for t in TrackDAO.get_all()]
    df = pd.DataFrame(rows).set_index("track_id", drop=False)
    df.index.name = None  # tránh mơ hồ "vừa là index vừa là cột" khi merge/groupby
    return df


def split_by_time(interactions=None, tracks=None):
    """Leave-last-invoice-out theo từng khách.

    test = hóa đơn cuối, val = hóa đơn kế cuối, train = phần còn lại.
    Khách có ít hơn MIN_INVOICES hóa đơn bị loại (không đủ để chia).
    """
    if interactions is None:
        interactions = InteractionDAO.get_all()
    if tracks is None:
        tracks = load_tracks()

    inv = (
        interactions[["customer_id", "invoice_id", "invoice_date"]]
        .drop_duplicates("invoice_id")
        .sort_values(["customer_id", "invoice_date", "invoice_id"])
    )
    # rank_from_end: 0 = hóa đơn cuối, 1 = kế cuối, ...
    inv["rank_from_end"] = inv.groupby("customer_id").cumcount(ascending=False)
    n_inv = inv.groupby("customer_id")["invoice_id"].transform("count")
    inv = inv[n_inv >= MIN_INVOICES]

    df = interactions.merge(inv[["invoice_id", "rank_from_end"]], on="invoice_id", how="inner")
    test = df[df.rank_from_end == 0].drop(columns="rank_from_end").reset_index(drop=True)
    val = df[df.rank_from_end == 1].drop(columns="rank_from_end").reset_index(drop=True)
    train = df[df.rank_from_end >= 2].drop(columns="rank_from_end").reset_index(drop=True)

    split = Split(train=train, val=val, test=test, tracks=tracks)
    check_no_leakage(split)
    return split


def check_no_leakage(split):
    """Các assert bắt buộc; sai là dừng ngay (AssertionError)."""
    tr, va, te = split.train, split.val, split.test

    # 1. Thứ tự thời gian theo từng khách: train < val < test
    g_tr = tr.groupby("customer_id").invoice_date.max()
    g_va = va.groupby("customer_id").invoice_date.agg(["min", "max"])
    g_te = te.groupby("customer_id").invoice_date.min()
    cust = g_tr.index
    assert set(cust) == set(g_va.index) == set(g_te.index), "Tập khách không đồng nhất giữa train/val/test"
    assert (g_tr.loc[cust] < g_va.loc[cust, "min"]).all(), "Rò rỉ: train không đứng trước val"
    assert (g_va.loc[cust, "max"] < g_te.loc[cust]).all(), "Rò rỉ: val không đứng trước test"

    # 2. Không có cặp (customer, track) nằm đồng thời ở train và val/test
    pairs = lambda d: set(zip(d.customer_id, d.track_id))
    assert not (pairs(tr) & pairs(te)), "Rò rỉ: cặp (customer, track) có ở cả train và test"
    assert not (pairs(tr) & pairs(va)), "Rò rỉ: cặp (customer, track) có ở cả train và val"
    assert not (pairs(va) & pairs(te)), "Rò rỉ: cặp (customer, track) có ở cả val và test"

    # 3. Hóa đơn không bị chia đôi giữa các tập
    ids = lambda d: set(d.invoice_id)
    assert not (ids(tr) & ids(va)) and not (ids(tr) & ids(te)) and not (ids(va) & ids(te)), \
        "Rò rỉ: một hóa đơn nằm ở nhiều tập"


def add_album_interactions(interactions, tracks):
    """Tương tác mức album: (customer, album) nếu khách mua >=1 track của album."""
    out = interactions.merge(tracks[["track_id", "album_id"]], on="track_id", how="left")
    return out.drop_duplicates(["customer_id", "album_id"])[["customer_id", "album_id"]].reset_index(drop=True)


def mark_item_cold(test, train_like):
    """Gắn cờ `item_cold` cho từng dòng test: track chưa ai mua trong train."""
    seen = set(train_like.track_id)
    out = test.copy()
    out["item_cold"] = ~out.track_id.isin(seen)
    return out


def candidate_tracks(customer_id, train_like, tracks):
    """Candidate = mọi track khách chưa mua trong tập train_like (không gợi ý lại)."""
    bought = set(train_like.loc[train_like.customer_id == customer_id, "track_id"])
    return tracks.index.difference(bought)


if __name__ == "__main__":
    s = split_by_time()
    print(f"khách={s.train.customer_id.nunique()}  train={len(s.train)}  val={len(s.val)}  test={len(s.test)}")
    cold = mark_item_cold(s.test, s.train_plus_val()).item_cold.mean()
    print(f"tỷ lệ dòng test là item-cold: {cold:.1%}")
    print("OK: các assert chống rò rỉ đều qua.")
