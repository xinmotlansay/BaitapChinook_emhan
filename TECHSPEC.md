# TECHSPEC — Hệ gợi ý Album & Track cho Customer (Chinook)

| | |
|---|---|
| Phiên bản | 1.0 (bản thiết kế hoàn chỉnh; chưa huấn luyện chính thức) |
| Ngày | 2026-10-22 |
| Loại dự án | Bài tập môn Học máy |
| Ngôn ngữ / UI | Python 3.11+, **PyQt6** + Qt Designer (`.ui`), giao diện tiếng Việt |
| Thư viện | pandas, numpy, scikit-learn, lightgbm, matplotlib, PyQt6, scipy |
| Dữ liệu | `databases/Chinook_Sqlite.sqlite` (chỉ đọc) |

---

## 1. Mục tiêu & phạm vi

**Mục tiêu:** Với mỗi customer, gợi ý Top-K **album** họ có khả năng mua tiếp. Với mỗi album được gợi ý, hiển thị các **track chưa mua**, xếp theo mức phù hợp do mô hình track dự đoán, kèm lý do gợi ý.

**Gồm:**
- Huấn luyện offline: làm sạch → chia train/test 80/20 theo thời gian → chọn tham số trên validation → huấn luyện → đánh giá → đóng gói model.
- Ứng dụng desktop một cửa sổ: gợi ý cho khách đã có và khách mới, xem kết quả đánh giá mô hình, chỉ nạp model đã đóng gói để suy luận.

**Không gồm:** API/server, deep learning, cập nhật dữ liệu thời gian thực, đóng gói `.exe`, danh sách track gợi ý độc lập (track chỉ xuất hiện trong album đã chọn), huấn luyện lại từ giao diện.

**Kỳ vọng:** dữ liệu nhỏ (59 customer) nên mục tiêu là *quy trình ML đúng, so sánh công bằng với baseline, kết luận trung thực kèm khoảng tin cậy*.

---

## 2. Phân tích dữ liệu (đã đo trên file thật)

| Hạng mục | Giá trị |
|---|---|
| Customer / Track / Album / Artist / Genre | 59 / 3.503 / 347 / 275 / 25 |
| Dòng hóa đơn (`InvoiceLine`) | 2.240, `Quantity` luôn = 1, không trùng (customer, track) |
| Cặp (customer, album) có mua | 1.301; trung bình 1,72 track mỗi cặp |
| Mỗi customer | 18–27 album (trung bình 22), 6–7 invoice |
| Thời gian | 2009-01-01 → 2013-12-22 |
| Track | 1.728/1.984 track bán chỉ 1 người mua; tối đa 2 người/track |
| Album | 304/347 album có người mua; **119 album có đúng 4 người mua, 39 album chỉ 1 người** |
| Mua lại cùng album ở hóa đơn khác | 0,2% số cặp (customer, album) |
| Album test: số track khách mua / kích thước album | 1,92 / 13,3 (tỉ lệ phủ trung bình 18%); 39,7% album test chỉ mua 1 track |
| Null | `Composer` 978 (28%); `Genre`, `Album` không null |
| Giá | 0,99 (95%) hoặc 1,99 (video/TV) |

**Kết luận thiết kế**
1. Ma trận customer×track thưa ~1% và gần như không chồng lấn → CF cấp track vô dụng, dùng content-based cho track. **Ma trận customer×album dày hơn nhiều** (khoảng 80% album có bán có 3–7 người mua) → CF cấp album dùng được.
2. Chỉ có tín hiệu implicit nhị phân (đã mua = 1).
3. Khách gần như không mua lại album đã mua (0,2%) → luôn loại album đã mua khỏi ứng viên; không chuyển sang gợi ý "mua nốt các track còn thiếu".
4. Track toàn catalog gần như không dự đoán được (Recall@10 khoảng 1–2%); đơn vị gợi ý chính là **album**, track chỉ xếp **trong album đã chọn**, và chỉ cần đo ở đúng bối cảnh đó (§7.3).
5. Chỉ 59 customer → mô hình nhẹ, chống overfit, báo cáo CI.

Làm sạch trên dữ liệu này không loại dòng nào; báo cáo làm sạch ghi nhận điều đó.

---

## 3. Yêu cầu

### 3.1 Chức năng

| ID | Yêu cầu |
|---|---|
| FR-01 | Đọc SQLite qua DAO; báo lỗi rõ nếu file sai hoặc thiếu bảng |
| FR-02 | Làm sạch dữ liệu (§4) và ghi báo cáo làm sạch (số dòng trước/sau, số dòng bị loại) |
| FR-03 | Tạo bảng tương tác customer–album và customer–track có thời gian mua |
| FR-04 | Chia train/test 80/20 theo thời gian từng customer, kiểm tra không rò rỉ bằng `assert` |
| FR-05 | Huấn luyện M1–M7 (album) và T0–T2 (track); chọn tham số trên validation |
| FR-06 | Đánh giá album trên test: Recall@K, Precision@K, NDCG@K, MAP@K, HitRate@K, Coverage, Diversity, CI 95% |
| FR-07 | Đánh giá track **trong album** (§7.3) |
| FR-08 | Kiểm tra độ nhạy theo điểm cắt, nhiều seed, so sánh cặp (§7.1) |
| FR-09 | Gợi ý Top-K album cho khách đã có (loại album đã mua), kèm lý do và track chưa mua xếp theo điểm |
| FR-10 | Gợi ý cho khách mới theo độ phổ biến của quốc gia và/hoặc một hoặc nhiều thể loại |
| FR-11 | Chọn mô hình album từ giao diện hoặc dòng lệnh; mặc định theo kết quả validation (§6.1) |
| FR-12 | Đóng gói model (thống kê, ma trận CF, LambdaRank album, LambdaRank track, bảng album) vào một file `.zip`; nạp lại để gợi ý mà không train lại |
| FR-13 | Xem lịch sử mua đầy đủ của khách đã có trong popup |
| FR-14 | Tab **Kết quả mô hình** trong giao diện: đọc các CSV trong `artifacts/` và hiển thị bảng, biểu đồ so sánh có CI |

### 3.2 Phi chức năng

| ID | Yêu cầu |
|---|---|
| NFR-01 | Tái lập: seed cố định (42), cấu hình trong `config.py` |
| NFR-02 | **Không rò rỉ dữ liệu:** mọi thống kê (popularity, hồ sơ, feature, TF-IDF, ma trận CF) chỉ tính từ train/cửa sổ lịch sử tương ứng; nhãn test không dùng để chọn tham số (kể cả `k`, `shrink`, `lambda` của CF) |
| NFR-03 | Huấn luyện chính dưới 1 phút; toàn bộ đánh giá phụ (độ nhạy điểm cắt, nhiều seed) dưới 5 phút; gợi ý một khách dưới 1 giây |
| NFR-04 | UI không treo khi nạp model và gợi ý (chạy nền bằng `QThread`) |
| NFR-05 | Code ML không phụ thuộc Qt (chạy được bằng `python -m ml.train`) |
| NFR-06 | Chỉ nạp file model do chính dự án tạo ra (pickle có thể chạy code tùy ý) |

---

## 4. Làm sạch & chuẩn bị dữ liệu

| Bước | Việc làm |
|---|---|
| 1 | Join `InvoiceLine → Invoice → Customer`, `Track → Album → Artist`, `Track → Genre` |
| 2 | Loại: dòng trùng `InvoiceLineId`, track không có trong catalog, `Quantity ≤ 0`, giá ≤ 0, ngày sai định dạng; ghi số dòng bị loại |
| 3 | Chuẩn hóa `InvoiceDate` sang `datetime`; strip tên artist và composer |
| 4 | `Composer` rỗng → `"Unknown"` và cờ `ComposerMissing` |
| 5 | Gộp thành tương tác customer–album: `NTracksBought`, `NTracksAlbum`, `Coverage`, `FirstPurchase`, `LastPurchase` |
| 6 | Gộp thành tương tác customer–track (kèm album của track) |
| 7 | Tạo bảng album (`NTracks`, `TotalMs`, `MeanPrice`, `ComposerMissingShare`, thể loại) |

Kết quả: báo cáo làm sạch (bảng `report`) được ghi vào log và `artifacts/cleaning_report.csv`.

---

## 5. Bài toán, chia dữ liệu, feature

### 5.1 Bài toán
Top-K recommendation với implicit feedback (đã mua = 1). Ứng viên là **toàn bộ album customer chưa mua trong train** (không lấy mẫu). Nhãn `y(u,a)=1` nếu u mua a trong giai đoạn đánh giá. Với track: ứng viên là các track của một album; nhãn `y=1` nếu customer mua track đó (§7.3).

### 5.2 Chia 80/20 theo thời gian, từng customer
1. Sắp album của customer theo `FirstPurchase`.
2. `n_train = round(0.8 × n)` album đầu → **train**; phần còn lại → **test**. Quy tắc này cho 1.041 dòng train và **260 dòng test** (trung bình 4,4 album/customer).
3. Từ train, 80% đầu → **sub_a**, 20% cuối → **sub_b**; từ sub, 80% đầu → **sub**, 20% cuối → **validation**. Tách theo thời gian, không trộn ngẫu nhiên.
4. Track đi theo album: track của một album thuộc đúng cửa sổ của album đó. Không có track test nào nằm trong lịch sử train của cùng customer.

### 5.3 Quy trình huấn luyện (album và track dùng chung)
- **Tuning:** huấn luyện trên cửa sổ (sub_a → nhãn sub_b), đánh giá trên cửa sổ (sub → nhãn validation). Chọn theo NDCG@10 trên validation.
  - LambdaRank (M5, T2): lưới 4 cấu hình (`num_leaves` ∈ {7, 15}, `n_estimators` ∈ {100, 300}).
  - M6: `k` ∈ {20, 50, tất cả}, `shrink` ∈ {0, 1}. M7: `lambda` ∈ {50, 200, 1000}.
- **Huấn luyện cuối:** dùng cả hai cửa sổ với tham số đã chọn.
- **Mẫu:** mỗi (customer, cửa sổ) là một nhóm LambdaRank gồm **mọi album chưa mua** (khoảng 330 ứng viên), nhãn dương là album của giai đoạn sau. Không lấy mẫu âm.
- **Ma trận/điểm CF** cho từng cửa sổ chỉ dựng từ lịch sử của cửa sổ đó.

### 5.4 Feature

**Album (`FEATURE_COLUMNS`, 14 cột):**
- Nội dung (cosine với hồ sơ customer): `genre_cos`, `artist_genre_cos`, `tfidf_cos`.
- Phổ biến: `pop_global`, `pop_country`.
- Nghệ sĩ: `artist_bought`, `artist_recency`.
- Catalog: `n_tracks`, `log_total_ms`, `mean_price`, `composer_missing`.
- Khách: `cust_n_albums`.
- CF: `cf_knn` (điểm M6), `cf_ease` (điểm M7), cả hai chuẩn hóa theo thứ hạng trong customer.

**Track (`TRACK_FEATURE_COLUMNS`, 13 cột):** `t_genre_cos`, `t_composer_cos`; `t_artist_bought`, `t_album_bought`; `t_album_content_cos`, `t_album_pop`; `t_track_pop`; `t_n_prior_buyers` (số customer khác đã mua track trong cửa sổ lịch sử), `t_album_unsold_share` (tỉ lệ track của album chưa ai mua); `t_unit_price`, `t_log_ms`, `t_composer_missing`; `t_cust_n_tracks`.

Hồ sơ customer chỉ được tính từ cửa sổ lịch sử tương ứng. Không dùng `Invoice.Total`, `InvoiceId`, hay thống kê nào của giai đoạn nhãn.

---

## 6. Mô hình

### 6.1 Album

| # | Mô hình | Cách tính điểm |
|---|---|---|
| M1 | Popularity | Số customer trong train đã mua album |
| M2 | Genre-profile | Cosine giữa thể loại album và hồ sơ thể loại customer; **tie-break theo M1** |
| M3 | Artist+Genre profile | Cosine giữa vector (nghệ sĩ + thể loại) và hồ sơ customer |
| M4 | Content TF-IDF | Cosine TF-IDF (nghệ sĩ, nhạc sĩ, thể loại) với hồ sơ customer |
| M5 | **Hybrid LightGBM (LambdaRank)** | Chấm điểm 14 feature (gồm điểm M2–M4, M6, M7) |
| M6 | **Item-kNN (CF)** | `score(u,a) = Σ_{b ∈ hồ sơ u} sim(a,b)`, `sim` = cosine giữa hai cột album của ma trận customer×album, giữ `k` láng giềng gần nhất, có `shrink` |
| M7 | **EASE (CF)** | Mô hình tuyến tính đóng: `B = −P / diag(P)`, `P = (RᵀR + λI)⁻¹`, `diag(B)=0`, `score = R·B` |

**Mô hình mặc định:** mô hình có NDCG@10 validation cao nhất giữa M5, M6, M7. Dự kiến M5 hoặc M6 (§7.2). Baseline mạnh nhất để so sánh cặp là M6 nếu M5 là mô hình mặc định, và M3 là mốc nội dung.

### 6.2 Track

| # | Mô hình | Cách tính điểm |
|---|---|---|
| T0 | Random | Xếp ngẫu nhiên trong album (seed cố định); mốc tham chiếu |
| T1 | Track popularity | Số customer trong train đã mua track |
| T2 | **Track LightGBM (LambdaRank)** | 13 feature track (§5.4); nhóm = (customer, album đã mua trong giai đoạn nhãn), ứng viên = mọi track của album. Dùng trong giao diện để xếp track của album |

Khi chọn album bằng mô hình nào, track vẫn dùng T2. Khách mới không có hồ sơ nên track xếp theo T2 với feature hồ sơ bằng 0 (nếu chưa tốt hơn T0 thì dùng `t_n_prior_buyers` tăng dần).

---

## 7. Đánh giá

### 7.1 Cách đo
- **Test chính:** chia 80/20 theo §5.2 (59 customer, 260 album test). Ứng viên là album chưa có trong train.
- Chỉ số chính: NDCG@10, Recall@10. Phụ: Precision, MAP, HitRate ở K ∈ {5, 10, 20}; Coverage@K; Diversity@K (1 − cosine thể loại trung bình trong danh sách).
- **CI 95%** bằng bootstrap theo customer (1.000 lần). So sánh cặp theo từng customer với M3 và với mô hình mặc định.
- **Nhiều seed (M5, 5 seed):** báo cáo khoảng và trung bình NDCG@10. M6, M7 xác định nên không cần seed.
- **Độ nhạy theo điểm cắt (báo cáo phụ, không dùng chọn mô hình):** lặp lại huấn luyện và đánh giá với tỉ lệ train ∈ {70%, 75%, 80%, 85%, 90%}, giữ nguyên tham số đã chọn ở 80/20. Mục đích: xem thứ hạng mô hình có ổn định không. Các test set chồng lấn nhau nên không coi là độc lập.
- Mọi tham số (kể cả `k`, `shrink`, `lambda`) chỉ chọn trên validation, không chọn trên test.

### 7.2 Kết quả

**Lần chạy đầu (chưa có feature CF, K = 10)**

| Mô hình | Recall@10 | NDCG@10 | HitRate@10 |
|---|---|---|---|
| M1 Popularity | 0,047 | 0,031 | 0,169 |
| M2 Genre-profile (chưa tie-break) | 0,000 | 0,000 | 0,000 |
| M3 Artist+Genre | 0,074 | 0,052 | 0,153 |
| M4 Content TF-IDF | 0,073 | 0,051 | 0,203 |
| M5 Hybrid LightGBM (không CF, lấy mẫu âm 30:1) | 0,100 | 0,085 | 0,322 |

So sánh cặp M5 với M3 (NDCG@10): +0,033, CI 95% [−0,013; 0,080], thắng 14 / thua 7: chưa thể kết luận M5 vượt M3. NDCG@10 của M5 qua 5 seed dao động 0,082–0,109 (trung bình ~0,091).

**Khảo sát CF (sơ bộ, chưa phải kết quả chính thức)**

Chạy một lần trên đúng cách chia 80/20 (1.041 train / 260 test) bằng `numpy` thuần. **Lần thử này có xem kết quả trên test và thử một lưới nhỏ tham số**, nên các số dưới đây chỉ để đặt kỳ vọng; kết quả chính thức chạy bằng pipeline đầy đủ với tham số chọn trên validation. Bản thử tái hiện M1 và M3 gần với lần chạy đầu nhưng không trùng hoàn toàn (khác chi tiết tie-break và cách dựng hồ sơ).

| Mô hình (bản thử) | Recall@10 | NDCG@10 | HitRate@10 |
|---|---|---|---|
| M1 Popularity | 0,039 | 0,027 | 0,136 |
| M3 Artist+Genre | 0,067 | 0,050 | 0,153 |
| M6 Item-kNN (6 cấu hình `k`, `shrink`) | 0,138–0,158 | 0,122–0,129 | 0,407–0,475 |
| M7 EASE (`lambda` 50 / 200 / 1000) | 0,135–0,138 | 0,115–0,119 | 0,424 |
| User-kNN (tham khảo) | 0,135 | 0,111 | 0,424 |
| Trung bình hạng M3 + M6 (tham khảo) | 0,122 | 0,093 | 0,356 |

Nhận xét: CF cấp album cao hơn rõ rệt các baseline nội dung và cao hơn M5 ở lần chạy đầu, và ít nhạy với tham số (NDCG@10 trong khoảng 0,115–0,129 ở mọi cấu hình đã thử). Trộn hạng với M3 làm giảm điểm, nên M5 phải được kiểm chứng trên validation chứ không mặc định tốt hơn M6.

### 7.3 Đánh giá track trong album
- **Đơn vị:** mỗi cặp (customer, album) mà customer mua trong giai đoạn test. Ứng viên = mọi track của album; nhãn = track được mua. Chỉ xét album có ≥ 6 track để top-5 có ý nghĩa.
- **Chỉ số:** HitRate@3, NDCG@5; CI 95% bootstrap theo customer. So với T0 (ngẫu nhiên).
- Chỉ số toàn catalog (Recall@10 khi xếp mọi track chưa mua) giữ làm **chỉ số phụ**. Lần chạy đầu: T1 Recall@10 0,000; T2 (chưa có feature `t_n_prior_buyers`) Recall@10 0,016, NDCG@10 0,015, HitRate@10 0,102.

**Khảo sát sơ bộ (267 cặp, chưa phải kết quả chính thức; số đo trên test, xem như giả thuyết cần kiểm chứng)**

| Cách xếp track trong album | HitRate@3 | NDCG@5 |
|---|---|---|
| T0 Ngẫu nhiên | 0,434 | 0,276 |
| T1 Nhiều người mua trước xếp trên | 0,221 | 0,176 |
| **Ít người mua trước xếp trên** | **0,554** | **0,400** |
| Track ngắn xếp trên | 0,363 | 0,260 |

T1 (popularity) kém hơn ngẫu nhiên, còn xếp ngược theo độ phổ biến thì tốt hơn rõ. Giả thuyết: mỗi track hiếm khi bị bán lặp (tối đa 2 người mua trên toàn dữ liệu) nên track chưa có ai mua dễ được mua hơn. Đây có thể là lý do T1 và T2 lần chạy đầu cho điểm thấp. T2 có thêm `t_n_prior_buyers` và `t_album_unsold_share` để học được quan hệ này; kết quả phải được xác nhận bằng validation, và bước phân tích lỗi cần kiểm tra xem hiệu ứng có do cách sinh dữ liệu hay không. Mẫu nhỏ (267 cặp), CI sẽ rộng.

### 7.4 Tiêu chí đạt
1. Mô hình album mặc định có NDCG@10 test cao hơn M1 và M3, và **CI 95% của chênh lệch cặp với M3 không chứa 0**. Nếu M5 và M6 không khác nhau có ý nghĩa thống kê, ghi nhận đúng như vậy và chọn mô hình đơn giản hơn (M6) làm mặc định.
2. Thứ hạng M5/M6/M3/M1 không đảo trong phần lớn các điểm cắt 70–90%.
3. T2 có NDCG@5 trong album cao hơn T0 với CI 95% của chênh lệch cặp không chứa 0; nếu không đạt, ghi nhận là chưa kết luận được.

### 7.5 Phân tích lỗi
Nội dung: theo thể loại, theo số giao dịch của customer, theo quốc gia; thiên lệch về album phổ biến (Coverage, tỉ lệ album gợi ý nằm trong top-20 phổ biến); kiểm tra giả thuyết "ít người mua trước" ở §7.3.

---

## 8. Cấu trúc dự án

```
BaitapChinook/
├─ databases/
│  └─ Chinook_Sqlite.sqlite
├─ connectors/
│  └─ connector.py          # kết nối SQLite dùng chung (ensureConnection / close)
├─ models/                  # lớp dữ liệu thuần (dataclass) + model đã đóng gói
│  ├─ customer.py           # Customer
│  ├─ album.py              # Album, Track
│  ├─ recommendation.py     # Recommendation (album, điểm, lý do, [(Track, điểm)])
│  └─ recommender.zip       # tạo bởi ml.train
├─ dal/                     # truy cập dữ liệu, nơi duy nhất viết SQL
│  ├─ CustomerDAO.py        # danh sách khách, quốc gia, lịch sử mua
│  └─ AlbumDAO.py           # catalog, dòng mua, track theo album, album/track đã mua
├─ ml/                      # học máy, không phụ thuộc Qt
│  ├─ data_prep.py          # làm sạch, tương tác, chia 80/20, cửa sổ, kiểm tra rò rỉ
│  ├─ features.py           # ProfileState, FEATURE_COLUMNS, TRACK_FEATURE_COLUMNS
│  ├─ recommenders.py       # M1–M7 (gồm CF), T0–T2, huấn luyện LambdaRank, MODEL_REGISTRY
│  ├─ evaluate.py           # metric, bootstrap CI, so sánh cặp, đánh giá track trong album, độ nhạy
│  ├─ train.py              # HUẤN LUYỆN
│  ├─ predict.py            # SUY LUẬN (Predictor), dòng lệnh
│  └─ package.py            # save_package / load_package
├─ ui/
│  ├─ RecommendMainWindow.ui     # thiết kế (Qt Designer)
│  ├─ RecommendMainWindow.py     # sinh bằng pyuic6, không sửa tay
│  ├─ RecommendMainWindowEx.py   # logic sự kiện, kế thừa Ui_MainWindow
│  ├─ HistoryDialog.py           # popup lịch sử mua
│  ├─ widgets.py                 # CheckComboBox (ô xổ xuống chọn nhiều)
│  └─ utils.py                   # showDataIntoTableWidget, drawGenreChart, relativeBars, Worker (QThread)
├─ test/
│  └─ run_app.py            # mở ứng dụng (xem §8.2)
├─ artifacts/               # metrics_album.csv, metrics_track_in_album.csv, paired.csv, seeds.csv, sensitivity_cut.csv, tuning.csv, cleaning_report.csv
├─ config.py                # đường dẫn, seed, tỉ lệ chia, lưới K, lưới tham số
├─ main.py                  # chạy ứng dụng
└─ requirements.txt
```

### 8.1 Quy tắc lập trình

**A. Học máy**
1. Khám phá dữ liệu trước khi huấn luyện: đếm dòng, giá trị thiếu, kiểu dữ liệu (ghi vào báo cáo làm sạch).
2. Feature khai báo tường minh (`FEATURE_COLUMNS`, `TRACK_FEATURE_COLUMNS`). Cột văn bản được mã hóa (TF-IDF, one-hot, đếm) trước khi đưa vào mô hình số.
3. Chia train/test trước khi fit; seed cố định.
4. Chỉ đánh giá trên test. Validation chỉ dùng để chọn tham số (gồm tham số CF), và nhãn validation không được dùng để huấn luyện cửa sổ đang đánh giá. Kết quả ở điểm cắt khác 80% chỉ để báo cáo độ nhạy.
5. Báo cáo chỉ số kèm CI, không chỉ in danh sách gợi ý.
6. Model lưu kèm tên feature. Khi suy luận, kiểm tra tên và thứ tự feature khớp với code; sai thì báo lỗi rõ.
7. Huấn luyện (`ml/train.py`) và suy luận (`ml/predict.py`) tách riêng. Suy luận chỉ nạp model đã đóng gói.
8. Đường dẫn tính bằng `Path(__file__)` trong `config.py`.
9. Model đóng gói dạng `.zip` chứa `meta.json` (đọc được) và `model.pkl` (pickle). Chỉ nạp file do dự án tạo ra (NFR-06).
10. Dùng `logging`, không dùng `print`, cho tiến trình huấn luyện và kết quả.
11. Chuỗi giao diện và thông báo hoàn toàn tiếng Việt; tên biến, hàm, class tiếng Anh; comment tiếng Việt có dấu.
12. Mọi mô hình album/track cài chung giao diện `fit(history)` và `score(customer, candidates)`, đăng ký trong `MODEL_REGISTRY` (`ml/recommenders.py`).

**B. Giao diện (PyQt6)**
1. Tách thiết kế và logic: `.ui` (Qt Designer) → `.py` (sinh bằng `pyuic6`, không sửa tay) → `*Ex.py` (logic).
2. Thứ tự khởi tạo: `__init__` → `setupUi()` → `setupTables()` → `setupSignalAndSlot()` → `setupChart()` → `loadModel()` → `loadFilters()` → `showWindow()`.
3. Lỗi hiển thị bằng `QMessageBox` (critical cho lỗi hệ thống/model, warning cho nhập sai). Mọi handler bọc `try/except`.
4. Kiểm tra đầu vào trước khi xử lý (chưa chọn khách, chưa chọn quốc gia hoặc thể loại cho khách mới).
5. Logic dữ liệu nằm ở DAO và `ml/`, UI chỉ gọi và hiển thị.
6. Bảng dùng chung `showDataIntoTableWidget(table, df)`.
7. SQL luôn dùng placeholder `?`.
8. Biểu đồ nhúng bằng `matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`; mỗi lần vẽ: `clear()` → vẽ → `tight_layout()` → `draw()`.
9. Tác vụ nặng (nạp model, gợi ý) chạy trong `QThread` với lớp `Worker` (`ui/utils.py`) và `pyqtSignal`; UI chỉ cập nhật trong slot nhận kết quả.
10. Mức phù hợp hiển thị theo `relativeBars` dạng thanh chấm + %: tương đối trong danh sách (100% cao nhất, 0% thấp nhất). Không hiển thị điểm thô vì thang điểm của mô hình không cố định.
11. Dùng enum đầy đủ của PyQt6 (ví dụ `Qt.AlignmentFlag.AlignCenter`, `QMessageBox.StandardButton.Ok`); gọi `app.exec()`; không cài đồng thời PyQt6 và PySide6 trong cùng môi trường.

**C. Kiểm tra**
1. Thư mục `test/` chỉ chứa file khởi chạy giao diện, không có unit test.
2. Chống rò rỉ được kiểm tra bằng `assert` trong `ml/data_prep.py` (không album test nào nằm trong train của cùng customer; mọi customer test có lịch sử train; ma trận CF của mỗi cửa sổ không chứa tương tác sau cửa sổ đó).

---

## 8.2 Thư mục `test/`

`test/run_app.py` mở ứng dụng để xem giao diện và thử gợi ý, không cần chuẩn bị thêm. File có đoạn thêm thư mục gốc vào `sys.path`, nên chạy từ đâu cũng được.

Trước khi nộp, kiểm tra trong môi trường ảo mới: chỉ cài `requirements.txt`, chạy `python -m ml.train` để tạo model, rồi mở `test/run_app.py`; giao diện phải hiện và gợi ý được cho khách đã có và khách mới.

---

## 9. Giao diện (PyQt6)

Một cửa sổ `RecommendMainWindow` gồm hai tab chính: **Gợi ý** và **Kết quả mô hình**.

### 9.1 Tab Gợi ý

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ Mô hình: [M5 Hybrid ▾]   Số album (K): [10 ▲▼]   [ Gợi ý ]        ● Model đã nạp │
├──────────────────────┬────────────────────────────────────────┬───────────────┤
│ [Khách đã có|Khách mới]│  TOP-K ALBUM                           │ HỒ SƠ KHÁCH   │
│ 🔍 Tìm theo tên/mã   │ # Album        Nghệ sĩ   Thể loại  Mức PH │ Tên, quốc gia │
│ ┌──────────────────┐ │ 1 Album A      Artist 1  Rock    ████ 100% │ 22 album đã mua│
│ │ 12  Nguyen Van A │ │ 2 Album B      Artist 2  Latin   ███░  82% │ Gu: Rock 41%… │
│ │ 15  John Smith   │ │ 3 …                                       │ ───────────── │
│ │ …                │ │───────────────────────────────────────────│ THỂ LOẠI ĐÃ MUA│
│ └──────────────────┘ │ VÌ SAO GỢI Ý ALBUM ĐANG CHỌN              │ (biểu đồ cột) │
│ (Khách mới: quốc gia │ • Giống với album X, Y bạn đã mua         │               │
│  + thể loại nhiều    │ • Thể loại Rock chiếm 41% lịch sử         │ [ Xem lịch sử │
│  lựa chọn)           │───────────────────────────────────────────│   mua (N) ]   │
│                      │ TRACK CHƯA MUA CỦA ALBUM                  │               │
│                      │ Track      Thể loại  Giá  Số người đã mua │               │
│                      │ …                                Mức PH   │               │
├──────────────────────┴────────────────────────────────────────┴───────────────┤
│ Trạng thái: Đã gợi ý 10 album · Mức phù hợp là tương đối trong danh sách        │
└───────────────────────────────────────────────────────────────────────────────┘
```

| Khu vực | Thành phần Qt | Nội dung |
|---|---|---|
| Thanh điều khiển | `QToolBar` hoặc `QHBoxLayout`: `QComboBox`, `QSpinBox`, `QPushButton` | Chọn mô hình (M1–M7, mặc định theo §6.1), K, nút Gợi ý (phím tắt Enter), chấm trạng thái model |
| Cột trái | `QTabWidget` | Tab **Khách đã có:** `QComboBox` cho phép gõ tìm (`QCompleter`, theo tên hoặc mã). Tab **Khách mới:** `QComboBox` quốc gia (có thể để trống), `CheckComboBox` thể loại, nút "Chọn tất cả" và "Xóa hết" |
| Bảng giữa trên | `QTableWidget` | Top-K album: #, album, nghệ sĩ, thể loại, mức phù hợp (thanh + %). Bấm một dòng để cập nhật hai khung bên dưới |
| Khung **Vì sao gợi ý** | `QTextBrowser` hoặc `QListWidget` | Lý do của album đang chọn. M6/M7: nêu 2–3 album đã mua giống nhất. M5: 2–3 feature đóng góp lớn nhất (`predict(..., pred_contrib=True)` của LightGBM). Khách mới: độ phổ biến theo quốc gia/thể loại |
| Bảng giữa dưới | `QTableWidget` | Track chưa mua của album đang chọn: track, thể loại, nhạc sĩ, thời lượng, giá, **số người đã mua trước**, mức phù hợp (T2) |
| Cột phải | `QGroupBox` + `FigureCanvasQTAgg` + `QPushButton` | Thẻ hồ sơ khách (quốc gia, số album, thể loại nổi bật); biểu đồ cột thể loại đã mua; nút **Xem lịch sử mua (N dòng)** mở `HistoryDialog`. Với khách mới: hồ sơ và biểu đồ trống, nút bị vô hiệu hóa |
| Thanh trạng thái | `QStatusBar` | Ghi chú về mức phù hợp, số album đã gợi ý, thời gian xử lý |

**Luồng gợi ý**
- *Khách đã có:* loại album đã mua (lấy toàn bộ từ DB) → chấm điểm bằng mô hình đã chọn → lý do → track chưa mua xếp bằng T2.
- *Khách mới:* chỉ dùng độ phổ biến. Album được giữ nếu thuộc ít nhất một thể loại đã chọn; theo quốc gia nếu có.
- Nút Gợi ý bị khóa và con trỏ chờ trong lúc worker chạy; kết quả quay về qua `pyqtSignal`.

### 9.2 Tab Kết quả mô hình (FR-14)

| Thành phần | Nội dung |
|---|---|
| Bảng chỉ số | Đọc `metrics_album.csv`: mô hình, Recall@10, NDCG@10, HitRate@10, CI 95% |
| Biểu đồ so sánh | Cột NDCG@10 kèm thanh lỗi CI cho M1–M7 (`FigureCanvasQTAgg`) |
| Bảng so sánh cặp | Đọc `paired.csv`: chênh lệch với M3 và với mô hình mặc định, CI, số khách thắng/thua |
| Độ nhạy điểm cắt | Biểu đồ đường NDCG@10 theo tỉ lệ train 70–90% cho từng mô hình |
| Track trong album | Bảng và biểu đồ HitRate@3, NDCG@5 của T0, T1, T2 |

Tab chỉ đọc file CSV; nếu thiếu file thì hiện thông báo "Chưa có kết quả, hãy chạy `python -m ml.train`".

### 9.3 Quy ước hiển thị
- Dùng kiểu `Fusion` để màu sắc và font đồng nhất giữa các máy; dùng `QSplitter` để người dùng kéo giãn các khung.
- Cột số căn phải, cột chữ căn trái; mọi thông báo bằng tiếng Việt.
- `Enter` sau khi chọn khách kích hoạt Gợi ý.

---

## 10. Kế hoạch & sản phẩm bàn giao

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 0 | Khảo sát dữ liệu, thử nghiệm sơ bộ, thiết kế (tài liệu này) | Xong |
| 1 | `ml/data_prep.py`: làm sạch, tương tác, chia 80/20 và kiểm tra rò rỉ | Chưa làm |
| 2 | `ml/evaluate.py`, baseline M1–M4, T0–T1 | Chưa làm |
| 3 | M6–M7 (CF), tuning trên validation | Chưa làm |
| 4 | Feature, M5 (14 feature), T2 (13 feature), `train.py`, `predict.py`, `package.py` | Chưa làm |
| 5 | Đánh giá: CI, so sánh cặp, nhiều seed, độ nhạy điểm cắt, đánh giá track trong album, phân tích lỗi | Chưa làm |
| 6 | Giao diện PyQt6 (tab Gợi ý, tab Kết quả mô hình) | Chưa làm |
| 7 | README, cập nhật mục 7 bằng kết quả chính thức, báo cáo ngắn | Chưa làm |

**Bàn giao:** mã nguồn, `databases/Chinook_Sqlite.sqlite`, `models/recommender.zip`, `requirements.txt`, file `.ui` và `.py` sinh ra, các CSV trong `artifacts/`, `README.md`, tài liệu này.

---

## 11. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Dữ liệu nhỏ → kết quả dao động | CI bootstrap, nhiều seed, so sánh cặp, độ nhạy điểm cắt |
| M5 không vượt M6 có ý nghĩa thống kê | Ghi nhận đúng khoảng tin cậy; chọn mô hình đơn giản hơn làm mặc định |
| Khảo sát sơ bộ đã xem test và thử vài tham số nên số liệu lạc quan | Tham số chỉ chọn trên validation trong pipeline chính; số sơ bộ ghi rõ là chưa chính thức; kết quả ít nhạy với tham số (§7.2) |
| Hiệu ứng "ít người mua trước" ở track có thể là đặc điểm của cách sinh dữ liệu, không tổng quát | Ghi nhận là giả thuyết, kiểm tra ở phân tích lỗi, không tuyên bố tổng quát hóa |
| Mẫu đánh giá track trong album nhỏ (267 cặp) | CI bootstrap theo customer; kết luận thận trọng |
| CF thiên về album phổ biến | Báo cáo Coverage và tỉ lệ album phổ biến trong danh sách (§7.5) |
| Rò rỉ dữ liệu | Thống kê và ma trận CF chỉ trên cửa sổ tương ứng; track đi theo album; `assert` trong `data_prep.py` |
| Overfit khi tune | Validation tách riêng, lưới nhỏ |
| Các điểm cắt khác 80% bị dùng để chọn mô hình | Chỉ báo cáo độ nhạy, không dùng chọn mô hình hay tham số |
| Pickle có thể chạy code tùy ý | Chỉ nạp file do dự án tạo ra (NFR-06) |
| Model cũ không khớp code | Kiểm tra tên feature khi nạp; báo lỗi và hướng dẫn chạy lại `ml.train` |
| Dữ liệu mới không tự vào model | Phải đưa dữ liệu vào DB và huấn luyện lại |
| Lẫn PyQt6 và PySide6 trong một môi trường | `requirements.txt` chỉ khai báo PyQt6 |

---

## 12. Các quyết định đã chốt

- Đơn vị gợi ý chính: **album**. Track được xếp hạng trong album đã chọn, không gợi ý độc lập, và được đánh giá trong đúng bối cảnh đó.
- **Giữ chia 80/20 theo thời gian từng customer** (`n_train = round(0.8 × n)`, 1.041 / 260); track đi theo album. Các điểm cắt khác chỉ để báo cáo độ nhạy.
- Mô hình: classical ML (baseline, CF Item-kNN và EASE) và LightGBM LambdaRank; không deep learning.
- Ứng viên là toàn bộ album chưa mua; không lấy mẫu âm.
- Mô hình album mặc định chọn theo NDCG@10 validation giữa M5, M6, M7; ưu tiên mô hình đơn giản khi không khác biệt có ý nghĩa.
- Huấn luyện offline bằng `python -m ml.train`. Ứng dụng chỉ suy luận từ `models/recommender.zip`.
- Đóng gói bằng zip chứa pickle.
- UI: **PyQt6** + Qt Designer, tiếng Việt, một cửa sổ hai tab (Gợi ý, Kết quả mô hình), `QTableWidget`, tác vụ nặng chạy `QThread`.
- Cấu trúc thư mục: `connectors / models / dal / ml / ui / test / artifacts`, mỗi màn hình có `.ui`, `.py` sinh tự động (`pyuic6`) và `*Ex.py`.
- Không đóng gói `.exe`.
- Mức phù hợp hiển thị tương đối trong danh sách, không hiển thị điểm thô.
