# Chinook RecSys — Hệ gợi ý Track / Album cho khách hàng

Bài tập môn học: với một khách hàng của cơ sở dữ liệu **Chinook**, hệ thống trả về **Top-K Track** và **Top-K Album** mà khách nhiều khả năng sẽ mua tiếp, kèm **lý do gợi ý**. Tín hiệu duy nhất là hành vi mua (implicit feedback); Playlist không được dùng.

Thiết kế chi tiết: [TECHSPEC.md](TECHSPEC.md).

## Chạy nhanh (không cần huấn luyện lại)

Repo đã kèm sẵn `models/chinook_recsys.zip` và `artifacts/*.csv`, nên có thể mở giao diện ngay.

1. **Python 3.11+.** Mở thư mục dự án trong PyCharm, tạo hoặc chọn interpreter (venv) rồi cài thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. **Mở giao diện:** chuột phải `test/run_app.py` → **Run** (không cần `main.py`, không cần cấu hình thêm).

Giao diện gồm 3 tab:
- **Gợi ý cho khách hàng:** chọn khách, Track/Album, mô hình (7 mô hình), Top-K rồi bấm *Gợi ý*. Bấm vào một dòng kết quả để xem **chi tiết lý do** (đối chiếu với lịch sử mua và các tín hiệu của mô hình Hybrid).
- **So sánh mô hình:** bảng và biểu đồ metric trên test kèm khoảng tin cậy bootstrap 95 %.
- **Thông tin dữ liệu & mô hình:** thống kê dữ liệu và metadata gói mô hình.

## Các lệnh (chạy tại thư mục gốc của dự án)

| Việc | Lệnh |
|---|---|
| Huấn luyện, đánh giá, sinh artifacts, đóng gói mô hình (~2 phút) | `python -m ml.train` |
| Chọn siêu tham số Hybrid trên tập val (~13 phút) | `python -m ml.tune` |
| Dự đoán dòng lệnh | `python -m ml.predict --customer 5 --kind track --k 10` |
| Kiểm tra split và chống rò rỉ | `python -m ml.data_prep` |
| Sinh lại `MainWindow.py` từ `.ui` | `pyuic6 ui/MainWindow.ui -o ui/MainWindow.py` |

`ml.predict` có thêm `--kind album` và `--model` (`M_hybrid`, `B0_random` … `B5_als`).

## Cách làm

**Dữ liệu.** 59 khách, 3 503 track, 2 240 tương tác (độ đặc ≈ 1,08 %). Chỉ 256 track được từ 2 khách trở lên mua, nên collaborative filtering thuần theo track gần như không có tín hiệu ⇒ mô hình chính là **hybrid** (nội dung + hành vi).

**Chia dữ liệu theo thời gian, từng khách** (leave-last-invoice-out): hóa đơn cuối = test, kế cuối = val, còn lại = train. `ml/data_prep.py` có assert bắt buộc (dừng nếu sai): train < val < test theo ngày của từng khách, không có cặp (khách, track) ở hai tập, không có hóa đơn nào bị chia đôi. Mọi thống kê (độ phổ biến, ma trận đồng xuất hiện, idf, hồ sơ khách) chỉ fit trên dữ liệu huấn luyện.

**Các mô hình**

| Mã | Mô hình |
|---|---|
| B0 | Random |
| B1 | Popularity |
| B2 | Thể loại/Nghệ sĩ ưa thích của khách × độ phổ biến |
| B3 | Item-kNN (cosine, đồng xuất hiện) trên mức album |
| B4 | Content-based: TF-IDF (thể loại, nghệ sĩ, định dạng, nhà soạn nhạc) |
| B5 | Matrix Factorization implicit: ALS viết bằng numpy |
| **M** | **Hybrid Ranker**: `HistGradientBoostingClassifier` trên 14 feature, kết hợp tín hiệu của B1–B5 |

Hybrid được huấn luyện bằng các "fold trong" theo thời gian (feature của mỗi fold chỉ tính từ hóa đơn trước hóa đơn đích) để tránh rò rỉ. Điểm album = điểm track tốt nhất của album. Siêu tham số chọn bằng `ml.tune` trên val và lưu ở `config.py`, kết quả tune ở `artifacts/tuning.csv`.

**Đánh giá.** Recall, HitRate, NDCG, MAP @ {5, 10, 20} tính từng khách rồi lấy trung bình; bootstrap 95 % CI (resample khách, 2 000 lần); paired bootstrap giữa M và từng baseline, kèm hiệu chỉnh Holm; 5 seed. Coverage và Novelty chỉ để chẩn đoán. Kết quả tách theo track/album và theo nhóm item *đã thấy* / *cold*.

## Kết quả trên tập test (Top-10, 59 khách, seed 42)

| Mức | Mô hình | Recall | NDCG | HitRate |
|---|---|---|---|---|
| Album | B0 Random | 0,027 | 0,012 | 0,085 |
| Album | B1 Popularity | 0,021 | 0,012 | 0,034 |
| Album | B2 Thể loại/Nghệ sĩ | 0,019 | 0,008 | 0,034 |
| Album | B3 Item-kNN | **0,103** | **0,084** | 0,288 |
| Album | B4 Content | 0,012 | 0,006 | 0,051 |
| Album | B5 ALS | 0,022 | 0,013 | 0,085 |
| Album | **M Hybrid** | 0,073 (CI 0,043–0,105) | 0,064 | **0,305** |
| Track | B0 Random | 0,004 | 0,005 | 0,034 |
| Track | B1–B5 | 0 | 0 | 0 |
| Track | **M Hybrid** | **0,011** (CI 0,004–0,023) | **0,013** | **0,085** |

Số liệu đầy đủ (mọi K, mọi nhóm, CI, so sánh cặp, trung bình ± std qua 5 seed) nằm trong `artifacts/`.

### Hạn chế (đọc trước khi diễn giải)

- **Mẫu rất nhỏ (59 khách):** khoảng tin cậy rộng. Ở mức album, Hybrid hơn rõ rệt Random, B1, B2, B4, B5 nhưng **không hơn được Item-kNN** (hiệu NDCG −0,019, CI chứa 0). Thứ hạng giữa các baseline không ổn định: B5 dẫn đầu ở val nhưng gần ngang Random ở test.
- **Mức track gần như không dự đoán được:** 65 % dòng test là *item-cold* (chưa ai mua ở train+val). Hybrid là mô hình duy nhất có tín hiệu (nhờ feature nội dung) nhưng recall chỉ ≈ 1 %. Sau hiệu chỉnh Holm, phần lớn so sánh cặp ở mức track chưa đạt p < 0,05.
- **Siêu tham số:** `neg_ratio=10` và `n_folds=3` là giá trị lớn nhất trong lưới tune; chọn cấu hình tốt nhất trong 72 cấu hình trên val có thể làm kết quả val lạc quan.
- **Chia theo thời gian từng khách** vẫn cho phép thống kê phổ biến dùng lượt mua của khách khác ở thời điểm muộn hơn val/test của khách đang xét. Đây là hệ quả của cách chia, không phải lỗi mã.
- `metrics.csv` và `paired.csv` lấy từ seed 42; `seeds.csv` là trung bình ± độ lệch chuẩn qua 5 seed.
- Giao diện chưa có test tự động. Bảng lịch sử trên UI gồm cả hóa đơn cuối (tập test) mà mô hình chưa thấy.

## Cấu trúc thư mục

```
├─ databases/      Chinook_Sqlite.sqlite (chỉ đọc, mở bằng mode=ro)
├─ connectors/     Connector.py
├─ dal/            CustomerDAO / TrackDAO / AlbumDAO / InteractionDAO  (nơi duy nhất có SQL)
├─ models/         entities.py (dataclass) + chinook_recsys.zip
├─ ml/             data_prep, features, recommenders, hybrid, evaluate, tune, train, predict, package
├─ ui/             MainWindow.ui → MainWindow.py (pyuic6, không sửa tay) → MainWindowEx.py (logic), utils, widgets
├─ test/           run_app.py
├─ artifacts/      metrics.csv, paired.csv, seeds.csv, tuning.csv
├─ config.py       đường dẫn, K, seed, siêu tham số
├─ requirements.txt
└─ TECHSPEC.md
```

Quy tắc phụ thuộc: `ui → ml.predict → dal → connectors`; `ml` không import Qt; SQL chỉ nằm trong `dal/`.

## Gói mô hình

`models/chinook_recsys.zip` gồm `model.joblib` (GBDT), `state.joblib` (thống kê, ma trận tương đồng, idf, nhân tố ALS) và `meta.json` (phiên bản thư viện, seed, danh sách feature, ngày train, metric). `load_package` từ chối nạp nếu danh sách feature không khớp mã hiện tại; khi đó chạy lại `python -m ml.train`. Gói được pickle bằng đúng phiên bản thư viện trong `requirements.txt`.
