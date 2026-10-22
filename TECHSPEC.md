# TECHSPEC — Hệ gợi ý Track / Album cho khách hàng (Chinook)

> Trạng thái: **v0.2** — đã chốt các lựa chọn chính ở mục 12; sẵn sàng viết code.

## 1. Mục tiêu

Với một khách hàng `c` bất kỳ, hệ thống trả về **Top-K Track** và **Top-K Album** mà `c` nhiều khả năng sẽ mua tiếp, kèm **lý do gợi ý** (vd. "cùng nghệ sĩ với bạn đã mua", "thể loại Rock bạn hay nghe").

Phạm vi bài tập môn học:
- Bài toán chọn: *Gợi ý track, album cho customer* (implicit feedback — hành vi mua).
- Có UI Qt để demo; thầy mở bằng PyCharm và chạy được ngay (không cần `main.py`).
- Có baseline → mô hình chính, đánh giá có khoảng tin cậy, có kiểm tra chống rò rỉ dữ liệu.

Ngoài phạm vi: huấn luyện online, A/B test, gợi ý cho khách mới hoàn toàn (cold-start user) ngoài một fallback đơn giản.

## 2. Dữ liệu (đã khảo sát trực tiếp file `Chinook_Sqlite.sqlite`)

| Bảng | Số dòng | Vai trò |
|---|---|---|
| Customer | 59 | người dùng (Country, City…) |
| Invoice | 412 | phiên mua, có `InvoiceDate` (2009-01-01 → 2013-12-22) |
| InvoiceLine | 2 240 | **tương tác** (Customer–Track), `Quantity` luôn = 1 |
| Track | 3 503 | item; có AlbumId, GenreId, MediaTypeId, Composer, Milliseconds, UnitPrice |
| Album / Artist / Genre | 347 / 275 / 25 | thuộc tính item |
| Playlist / PlaylistTrack | 18 / 8 715 | **không dùng** (playlist của hệ thống, không cá nhân hóa) |

**Những con số quyết định thiết kế:**
- Ma trận Customer×Track chỉ ~1.1 % có giá trị (mỗi khách ~38 track khác nhau trên 3 503).
- Chỉ 1 984 track được mua; **chỉ 256 track được ≥2 khách mua** → collaborative filtering thuần theo track gần như không có tín hiệu.
- Chỉ 304 album có track được mua → ở mức **album/artist/genre** dữ liệu "đặc" hơn nhiều.
- Mỗi khách ≈ 7 hóa đơn → đủ để chia theo thời gian.

⇒ **Kết luận thiết kế:** mô hình chính phải là **hybrid (nội dung + hành vi)**, không thể chỉ dựa vào CF thuần. Đây cũng là điểm đáng phân tích trong báo cáo.

## 3. Định nghĩa bài toán

- **Tương tác implicit:** `(customer, track)` tồn tại nếu có InvoiceLine. Không dùng `Quantity` (hằng 1).
- **Tương tác mức album:** `(customer, album)` nếu khách mua ≥1 track thuộc album đó (bảng riêng, sinh từ bảng track).
- **Candidate set:** mọi track/album **chưa** nằm trong tập train của khách (không gợi ý lại thứ đã mua).
- **Đầu ra:** danh sách xếp hạng `[(item_id, score, reason)]`.

## 4. Chia dữ liệu & chống rò rỉ

Chia **theo thời gian, theo từng khách** (leave-last-invoice-out):

| Tập | Gồm |
|---|---|
| test | hóa đơn **cuối** của mỗi khách |
| val | hóa đơn **kế cuối** (dùng chọn siêu tham số, trọng số hybrid) |
| train | các hóa đơn còn lại |

Khi huấn luyện lần cuối: train+val → đánh giá trên test.

**Assert bắt buộc** (trong `data_prep.py`, fail là dừng):
1. `max(date_train[c]) < min(date_val[c]) < min(date_test[c])` với mọi khách.
2. Không có cặp (customer, track) nào xuất hiện đồng thời trong train và test (đã loại ở candidate set).
3. Độ phổ biến, ma trận đồng xuất hiện, hồ sơ khách, TF-IDF idf… **chỉ fit trên train**.
4. Không dùng `Invoice.Total`, `InvoiceDate` của test làm feature.
5. Thuộc tính tĩnh của item (genre, artist, giá, độ dài) được phép dùng — không phụ thuộc thời gian.

Lưu ý khi báo cáo: nhiều track trong test là **chưa ai mua trong train** (item-cold) — nêu rõ và báo cáo thêm metric ở mức album/artist/genre.

## 5. Mô hình (baseline → chính)

| Mã | Mô hình | Ghi chú |
|---|---|---|
| B0 | Random | sàn so sánh |
| B1 | Popularity | số khách mua trong train |
| B2 | Genre/Artist-Popularity cá nhân hóa | điểm = tần suất genre/artist của khách × độ phổ biến |
| B3 | Item-kNN (cosine, đồng xuất hiện) | chạy trên mức album để đỡ thưa |
| B4 | Content-based | vector TF-IDF/one-hot (genre, artist, media type, composer) → hồ sơ khách = trung bình item đã mua → cosine |
| B5 | Matrix Factorization implicit (ALS hoặc BPR) | triển khai bằng numpy/`implicit` — xem mục 12 |
| **M** | **Hybrid Ranker (mô hình chính)** | học xếp hạng điểm (pointwise) bằng `HistGradientBoostingClassifier` trên feature kết hợp tín hiệu của B1–B5 |

**Mô hình chính M** — huấn luyện trên cặp (customer, candidate track):
- Positive = track mua ở hóa đơn "tương lai" trong cửa sổ huấn luyện; Negative = lấy mẫu ngẫu nhiên từ candidate (tỷ lệ 1:N, N cấu hình).
- `FEATURE_COLUMNS` (khai báo trong `features.py`), dự kiến:
  `pop_track, pop_album, pop_artist, genre_affinity, artist_affinity, album_affinity, content_cosine, knn_score, mf_score, country_genre_pop, unit_price, milliseconds_log, same_media_type_ratio, days_since_last_purchase`
- Gợi ý **Album** = gộp điểm của các track trong album (max + mean, cấu hình) rồi xếp hạng; so sánh với mô hình xếp hạng riêng trên bảng album (B3/B4 mức album).
- Cold-start khách (không có lịch sử): fallback B1 + lọc theo Country.

Đóng gói: `models/chinook_recsys.zip` gồm `model.joblib`, `state.joblib` (ma trận, idf, bảng phổ biến), `meta.json` (phiên bản thư viện, seed, FEATURE_COLUMNS, ngày train, metric). `load_package` kiểm tra khớp FEATURE_COLUMNS.

## 6. Đánh giá

- **Metric** @K ∈ {5, 10, 20}: Recall, HitRate, NDCG, MAP; thêm Coverage và Novelty (diagnostic).
- Tính **từng khách rồi trung bình** (59 khách → mẫu nhỏ, bắt buộc CI).
- **Bootstrap 95 % CI** (resample khách, 2 000 lần).
- **So sánh cặp:** paired bootstrap giữa M và từng baseline (hiệu metric, CI, p-value); ghi rõ không hiệu chỉnh đa so sánh hoặc dùng Holm.
- **Nhiều seed** (5 seed: lấy mẫu âm, khởi tạo MF) → báo mean ± std.
- Báo cáo riêng: (a) track-level, (b) album-level, (c) nhóm test "item đã thấy" vs "item cold".
- Artifacts xuất ra `artifacts/`: `metrics.csv`, `paired.csv`, `tuning.csv`, `seeds.csv`.

## 7. Kiến trúc & cấu trúc thư mục

```
ChinookRecSys/
├─ databases/        Chinook_Sqlite.sqlite (chỉ đọc; connector mở bằng mode=ro)
├─ connectors/       Connector.py             1 class Connector (sqlite3 URI ro)
├─ dal/              CustomerDAO / TrackDAO / AlbumDAO / InteractionDAO  (@staticmethod, nơi duy nhất có SQL)
├─ models/           dataclass: Customer, Track, Album, Recommendation  + chinook_recsys.zip
├─ ml/               data_prep.py, features.py, recommenders.py, hybrid.py,
│                    evaluate.py, train.py, predict.py, package.py   (KHÔNG import Qt)
├─ ui/               MainWindow.ui → MainWindow.py (pyuic, tự sinh) → MainWindowEx.py (logic), utils.py, widgets.py
├─ test/             run_app.py   ← chuột phải → Run trong PyCharm
├─ artifacts/        metrics.csv, paired.csv, tuning.csv, seeds.csv
├─ config.py         đường dẫn, K, seed, tỷ lệ âm, siêu tham số
├─ requirements.txt  (ghim ==)
├─ README.md
└─ TECHSPEC.md
```

Quy tắc phụ thuộc: `ui → ml.predict → dal → connectors`; `ml` không import `ui`; dataclass trong `models/` không có hành vi; SQL **chỉ** ở `dal/`.
`test/run_app.py` tự thêm project root vào `sys.path` để chạy được từ PyCharm mà không cần cấu hình thêm.

## 8. Giao diện Qt (đề xuất)

Cửa sổ chính, 3 tab:

**Tab 1 — Gợi ý cho khách hàng** *(tab chính khi demo)*
```
┌ Khách hàng: [ComboBox 59 khách ▾]  Quốc gia: Brazil   Đã mua: 38 track
│ Đối tượng: (•) Track ( ) Album   Mô hình: [Hybrid ▾]   Top-K: [10]  [Gợi ý]
├───────────────────────────────┬──────────────────────────────────────┤
│ Lịch sử mua (QTableView)      │ Kết quả gợi ý (QTableView)           │
│ Track | Album | Artist | Ngày │ # | Track/Album | Artist | Điểm | Lý do│
├───────────────────────────────┴──────────────────────────────────────┤
│ Biểu đồ: cơ cấu thể loại khách đã mua (pie/bar) │ Trạng thái / log   │
└──────────────────────────────────────────────────────────────────────┘
```

**Tab 2 — So sánh mô hình:** bảng `metrics.csv` (mean + CI) và biểu đồ cột có error bar theo metric/K; chọn Track/Album.

**Tab 3 — Thông tin dữ liệu & mô hình:** thống kê dữ liệu (độ thưa, phân bố), metadata gói mô hình (ngày train, seed, thư viện).

Chi tiết kỹ thuật: chạy dự đoán trên `QThread`/`QRunnable` để UI không đơ; biểu đồ bằng `matplotlib` nhúng `FigureCanvasQTAgg` (hoặc Qt Charts — xem câu hỏi); `widgets.py` chứa widget dùng lại (bảng có sort, canvas biểu đồ); `utils.py` chứa hàm format/thông báo lỗi.

## 9. Luồng chạy

| Việc | Lệnh (chạy tại root dự án) |
|---|---|
| Huấn luyện + đánh giá + sinh artifacts + đóng gói | `python -m ml.train` |
| Dự đoán dòng lệnh | `python -m ml.predict --customer 5 --kind track --k 10` |
| Mở UI | chạy `test/run_app.py` |
| Sinh `.py` từ `.ui` | `pyuic6 ui/MainWindow.ui -o ui/MainWindow.py` |

Repo nộp kèm sẵn `models/chinook_recsys.zip` và `artifacts/*.csv` để thầy mở UI chạy ngay mà không cần train lại.

## 10. Công nghệ & phiên bản (sẽ ghim `==` ở `requirements.txt`)

Python 3.11+; numpy, pandas, scipy, scikit-learn, joblib, matplotlib, PyQt6 (hoặc PySide6). Cố gắng **không** dùng thư viện nặng/khó cài (vd. `implicit` cần compiler) — ALS/BPR tự viết bằng numpy nếu cần.

## 11. Rủi ro & quyết định

| Rủi ro | Quyết định |
|---|---|
| Dữ liệu quá thưa, 59 khách | hybrid + metric mức album/genre + CI bootstrap; trình bày trung thực giới hạn |
| Item-cold trong test | báo cáo tách nhóm; M dùng feature nội dung nên vẫn xếp hạng được |
| Rò rỉ thời gian | assert ở `data_prep.py`, unit-check trong `test/` |
| Pyuic sinh file bị sửa tay | quy ước: không sửa `X.py`, mọi logic ở `XEx.py` |
| Thầy không có ĐK cài | `requirements.txt` ghim, README hướng dẫn PyCharm interpreter, kèm model zip |

## 12. Quyết định đã chốt với người dùng
- Qt: **PyQt6** (`pyuic6`).
- Mô hình chính: **Hybrid Ranker GBDT** (`HistGradientBoostingClassifier`).
- Ngôn ngữ: UI, README, TECHSPEC, comment bằng **tiếng Việt**; tên biến/hàm bằng tiếng Anh.
- Playlist/PlaylistTrack: **không dùng**, chỉ dùng tín hiệu mua hàng.

## 13. Còn mở (sẽ mặc định nếu không có ý kiến)
- Chia dữ liệu: leave-last-invoice-out theo thời gian (mục 4).
- Biểu đồ: matplotlib nhúng Qt (mục 8).
- MF (B5): tự viết ALS bằng numpy, không dùng `implicit`.
