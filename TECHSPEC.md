# TECHSPEC — Dự báo doanh thu (Tổng, theo Album, theo Track) — Chinook

| | |
|---|---|
| Phiên bản | 1.0 (bản thiết kế, chưa huấn luyện chính thức) |
| Ngày | 2026-10-22 |
| Loại dự án | Bài tập môn Học máy |
| Ngôn ngữ / UI | Python 3.11+, PySide6 + Qt Designer (`.ui`), giao diện tiếng Việt |
| Thư viện | pandas, numpy, scikit-learn, lightgbm, statsmodels, matplotlib, PySide6, scipy |
| Dữ liệu | `databases/Chinook_Sqlite.sqlite` (chỉ đọc) |

---

## 1. Mục tiêu & phạm vi

**Mục tiêu:** Dự báo doanh thu **12 tháng tiếp theo** ở ba cấp, và bảo đảm ba cấp khớp nhau (tổng track = tổng album = tổng toàn cửa hàng):

| Cấp | Đối tượng dự báo | Đơn vị thời gian |
|---|---|---|
| Tổng | Doanh thu toàn cửa hàng | Từng tháng (h = 1…12) |
| Album | Doanh thu của từng album | Tổng 12 tháng |
| Track | Doanh thu của từng track | Tổng 12 tháng |

Doanh thu = `UnitPrice × Quantity` của `InvoiceLine`. Tổng này trùng `Invoice.Total` (đã kiểm: 0 hóa đơn lệch).

**Gồm:**
- Huấn luyện offline: làm sạch → dựng chuỗi → chia theo *origin* thời gian → chọn tham số trên validation → huấn luyện → đánh giá → hòa giải phân cấp → đóng gói model.
- Ứng dụng desktop một màn hình: xem lịch sử và dự báo kèm khoảng dự báo ở cả ba cấp, chỉ nạp model đã đóng gói.

**Không gồm:** API/server, deep learning, cập nhật dữ liệu thời gian thực, đóng gói `.exe`, dự báo theo quốc gia hoặc khách hàng, dự báo album/track mới chưa có trong catalog, huấn luyện lại từ giao diện.

**Kỳ vọng:** dữ liệu nhỏ, rất thưa và có cấu trúc sinh khá đều (§2). Mục tiêu là *quy trình dự báo đúng, so sánh công bằng với baseline đơn giản, kết luận trung thực* (kể cả kết luận "không mô hình nào vượt baseline" ở cấp Tổng), không phải sai số thấp bằng mọi giá.

---

## 2. Phân tích dữ liệu (đã đo trên file thật)

| Hạng mục | Giá trị |
|---|---|
| Hóa đơn / dòng hóa đơn | 412 / 2.240 (`Quantity` luôn = 1) |
| Tổng doanh thu | 2.328,60 |
| Thời gian | 2009-01-01 → 2013-12-22 (60 tháng) |
| Doanh thu theo năm | 449,46 · 481,45 · 469,58 · 477,53 · 450,58 (2009→2013) |
| Doanh thu tháng | trung bình 38,81; độ lệch chuẩn 4,74 (CV 12,2%); **72% số tháng đúng bằng 37,62** |
| Hóa đơn mỗi tháng | 5–7 (trung bình 6,87) |
| Hóa đơn mỗi customer | 6–7; kích thước (số dòng) theo khuôn cố định 1, 2, 2, 4, 6, 9, 14 = 38 dòng |
| Giá dòng | 0,99 (2.129 dòng) hoặc 1,99 (111 dòng video/TV) |
| Track có bán / tổng | 1.984 / 3.503; 1.728 track bán 1 lần, 256 track bán 2 lần |
| Doanh thu mỗi track | 0,99 (1.633) · 1,98 (248) · 1,99 (95) · 3,98 (8) |
| Khoảng cách giữa hai lần bán cùng một track | tối thiểu 473 ngày, trung vị 934 ngày |
| Album có bán / tổng | 304 / 347; doanh thu mỗi album: trung bình 7,66, trung vị 6,93, lớn nhất 35,82 |
| Tương quan doanh thu album với số track / số người mua | 0,82 / 0,83 |
| Độ thưa | album–tháng: 5,3% ô khác 0; track–tháng: 1,07% ô khác 0 |
| Album có doanh thu 0 trong một năm | khoảng 30% (mỗi năm) |
| Tương quan doanh thu album năm n và n+1 | 0,13 · 0,41 · 0,30 · 0,12 (yếu) |
| Năm 2013: track có bán / chưa từng bán trước đó | 442 / 313 (71%); 29% doanh thu track 2013 đến từ track đã bán trước đó |
| Năm 2013: album có bán đã từng bán trước đó | 205 / 205 (100%) |
| Null | Không có null ảnh hưởng doanh thu |

**Kết luận thiết kế**
1. **Cấp Tổng gần như là hằng số** (~37,62 mỗi tháng, không có mùa vụ rõ: trung bình theo tháng trong năm chỉ dao động 37,25–40,22; tự tương quan lag 1 = 0,31, lag 12 = −0,12). Các đỉnh đến từ dòng giá 1,99. Baseline đơn giản rất mạnh; mô hình phức tạp khó thắng. Cần báo cáo điều này thay vì ép mô hình học máy "thắng".
2. **Cấp Album là nơi có tín hiệu thật:** doanh thu album gắn chặt với số track và số người mua; 100% album bán năm 2013 đã có lịch sử trước đó. Chuỗi tháng quá thưa nên **dự báo theo cửa sổ 12 tháng**, không theo tháng.
3. **Cấp Track gần như không có chuỗi thời gian:** mỗi track bán 1–2 lần trong 5 năm, 71% track bán năm 2013 chưa từng bán. Dự báo track = *phân bổ từ album* + xác suất bán × giá, không dựng mô hình chuỗi thời gian riêng cho từng track.
4. Sai số tuyệt đối trung bình (MAE) **gây hiểu lầm** ở cấp Track: dự báo toàn 0 cho MAE tốt nhất (xem §7.3). Phải dùng RMSE, độ lệch Poisson/Tweedie, WAPE theo nhóm và chỉ số xếp hạng.
5. Catalog không có ngày phát hành: coi mọi album/track có sẵn suốt 5 năm. Ghi nhận là hạn chế.
6. Tháng 12/2013 chỉ có dữ liệu đến 22/12 nhưng doanh thu (38,62) vẫn ở mức bình thường; giữ nguyên và ghi chú trong báo cáo làm sạch.

---

## 3. Yêu cầu

### 3.1 Chức năng

| ID | Yêu cầu |
|---|---|
| FR-01 | Đọc SQLite qua DAO; báo lỗi rõ nếu file sai hoặc thiếu bảng |
| FR-02 | Làm sạch dữ liệu (§4), đối chiếu doanh thu dòng với `Invoice.Total`, ghi báo cáo làm sạch |
| FR-03 | Dựng chuỗi tổng theo tháng (đủ 60 tháng, tháng trống = 0) và bảng doanh thu album/track theo cửa sổ 12 tháng tại từng origin |
| FR-04 | Chia theo origin thời gian (§5.2), kiểm tra không rò rỉ bằng `assert` |
| FR-05 | Huấn luyện F1–F6 (Tổng), A1–A5 (Album), K1–K4 (Track); chọn tham số trên validation |
| FR-06 | Đánh giá trên test theo chỉ số từng cấp (§7.1), kèm CI 95% và so sánh cặp với baseline mạnh nhất |
| FR-07 | Hòa giải phân cấp R1–R3 (§6.4); báo cáo độ lệch trước/sau hòa giải |
| FR-08 | Sinh khoảng dự báo 80% (§7.1) cho cả ba cấp |
| FR-09 | Chọn mô hình từng cấp và phương pháp hòa giải từ giao diện hoặc dòng lệnh; mặc định là mô hình thắng trên validation |
| FR-10 | Đóng gói model (mô hình ba cấp, phân vị sai số, bảng album/track, chuỗi tháng) vào một file `.zip`; nạp lại để dự báo mà không train lại |
| FR-11 | Xem lịch sử doanh thu (đồ thị + bảng) của tổng, từng album, từng track |
| FR-12 | Dự báo tương lai: huấn luyện lại trên mọi origin có nhãn đầy đủ và dự báo 12 tháng sau ngày cuối của dữ liệu (2014) |

### 3.2 Phi chức năng

| ID | Yêu cầu |
|---|---|
| NFR-01 | Tái lập: seed cố định (42), cấu hình trong `config.py` |
| NFR-02 | **Không rò rỉ dữ liệu:** mọi feature tại origin `o` chỉ dùng giao dịch có `InvoiceDate ≤ o`; nhãn có cửa sổ `(o, o+12 tháng]` không được lọt vào lịch sử của bất kỳ dòng nào dùng để đánh giá; nhãn test không dùng để chọn tham số |
| NFR-03 | Huấn luyện toàn bộ dưới 1 phút; dự báo một album/track dưới 1 giây |
| NFR-04 | UI không treo khi nạp model và dự báo |
| NFR-05 | Code ML không phụ thuộc Qt (chạy được bằng `python -m ml.train`) |
| NFR-06 | Chỉ nạp file model do chính dự án tạo ra (pickle có thể chạy code tùy ý) |
| NFR-07 | **Nhất quán phân cấp:** sau hòa giải, `Σ track = album` và `Σ album = Tổng` (sai lệch tuyệt đối < 1e-6) |
| NFR-08 | Dự báo không âm ở mọi cấp |

---

## 4. Làm sạch & chuẩn bị dữ liệu

| Bước | Việc làm |
|---|---|
| 1 | Join `InvoiceLine → Invoice`, `Track → Album → Artist`, `Track → Genre`, `Track → MediaType` |
| 2 | Loại: dòng trùng `InvoiceLineId`, track không có trong catalog, `Quantity ≤ 0`, giá ≤ 0, ngày sai định dạng; ghi số dòng bị loại (dữ liệu này dự kiến không loại dòng nào) |
| 3 | Chuẩn hóa `InvoiceDate` sang `datetime`; tính `Revenue = UnitPrice × Quantity` |
| 4 | Đối chiếu `Σ Revenue` theo hóa đơn với `Invoice.Total`; ghi số hóa đơn lệch (đã đo: 0) |
| 5 | Dựng chuỗi Tổng theo tháng `MS`, đủ 60 tháng, tháng không có bán = 0 |
| 6 | Tạo bảng catalog album (`NTracks`, `TotalMs`, `MeanPrice`, `VideoShare`, thể loại chính, nghệ sĩ) và catalog track |
| 7 | Với mỗi origin `o`, dựng bảng (album, o) và (track, o): feature từ lịch sử `≤ o` và nhãn `y` = doanh thu trong `(o, o+12 tháng]` (§5) |

Kết quả: báo cáo làm sạch (bảng `report`) được ghi vào log khi huấn luyện.

---

## 5. Bài toán, chia dữ liệu, feature

### 5.1 Bài toán
- **Tổng:** dự báo chuỗi `R_t` (t = tháng) cho 12 tháng tiếp theo từ lịch sử đến origin.
- **Album/Track:** với origin `o`, dự báo `y(e, o) = Σ doanh thu của e trong (o, o+12 tháng]` cho mọi `e` trong catalog (kể cả album/track chưa từng bán). Mục tiêu không âm, dạng *zero-inflated* (≈ 30% album và ≈ 87% track-origin có `y = 0`).

### 5.2 Chia theo origin thời gian (rolling-origin)

| Vai trò | Origin | Cửa sổ nhãn |
|---|---|---|
| Train (tuning) | 2009-12-31, 2010-06-30, 2010-12-31 | đến hết 2011-12-31 |
| **Validation** | 2011-12-31 | 2012-01 → 2012-12 |
| Train (cuối) | 2009-12-31, 2010-06-30, 2010-12-31, 2011-06-30, 2011-12-31 | đến hết 2012-12-31 |
| **Test** | 2012-12-31 | 2013-01 → 2013-12 |
| Sản xuất (FR-12) | thêm 2012-06-30, 2012-12-31 vào train; origin dự báo 2013-12-31 | 2014 |

- Origin cách nhau 6 tháng, nhãn 12 tháng nên cửa sổ nhãn của các dòng train chồng lấn nhau. Chấp nhận được vì mọi nhãn train đều kết thúc **trước** origin của tập được đánh giá. Ghi nhận mức tự tương quan này khi đọc CI.
- Số dòng: album 347 × 5 = 1.735; track 3.503 × 5 = 17.515.
- Cấp Tổng: huấn luyện trên 48 tháng 2009-01→2012-12, kiểm tra trên 12 tháng 2013. Chọn tham số bằng backtest origin 2010-12-31 và 2011-12-31 (mỗi lần dự báo 12 tháng).
- `assert` trong `data_prep.py`: (1) mọi feature của dòng (e, o) dùng ngày ≤ o; (2) ngày cuối của cửa sổ nhãn train ≤ origin của tập đánh giá; (3) không có nhãn test trong bảng train.

### 5.3 Quy trình huấn luyện LightGBM (album và track dùng chung quy trình)
- **Tuning:** huấn luyện trên 3 origin train (tuning), đánh giá trên validation (origin 2011-12-31). Lưới 4 cấu hình: `num_leaves` ∈ {3, 7}, `n_estimators` ∈ {100, 300}. Chọn theo RMSE trên validation.
- **Huấn luyện cuối:** dùng 5 origin với tham số đã chọn.
- **Mục tiêu học:** `objective = tweedie` (`tweedie_variance_power = 1.3`), phù hợp doanh thu không âm có nhiều số 0. Các tham số khác cố định (`learning_rate = 0.05`, `min_child_samples = 20`, `seed = 42`).
- Không lấy mẫu âm (khác đề gợi ý): mọi album/track của catalog đều là một dòng, nhãn là doanh thu thật.

### 5.4 Feature

**Album (`FEATURE_COLUMNS`, 13 cột):**
- Tĩnh theo catalog: `n_tracks`, `log_total_ms`, `mean_price`, `video_share`.
- Lịch sử đến origin: `rev_all`, `rev_12`, `rev_6` (doanh thu toàn bộ / 12 tháng / 6 tháng gần nhất), `n_buyers`, `months_since_last_sale`, `rev_rate` (doanh thu quy năm = `rev_all / số tháng lịch sử × 12`), `sold_track_share` (tỉ lệ track của album đã bán ít nhất một lần).
- Ngữ cảnh: `artist_other_rev` (doanh thu các album khác cùng nghệ sĩ), `genre_rev_share` (tỉ trọng doanh thu của thể loại chính).

**Track (`TRACK_FEATURE_COLUMNS`, 12 cột):**
- Tĩnh: `t_unit_price`, `t_log_ms`, `t_is_video`, `t_position_in_album`.
- Lịch sử: `t_sold_before`, `t_buyers`, `t_months_since_sold`.
- Từ album chứa track: `t_album_n_tracks`, `t_album_rev_12`, `t_album_sold_share`, `t_album_rev_rate`.
- Ngữ cảnh: `t_genre_rev_share`.

**Tổng (cho F6):** `lag1`, `lag2`, `lag12`, `roll_mean_3`, `roll_mean_12`, `month`.

Không dùng `InvoiceId`, `CustomerId` làm feature, không dùng bất kỳ thống kê nào của giai đoạn nhãn.

---

## 6. Mô hình

### 6.1 Tổng (chuỗi tháng, h = 1…12)

| # | Mô hình | Cách tính |
|---|---|---|
| F1 | Mean | Trung bình toàn bộ tháng train |
| F2 | Median | Trung vị toàn bộ tháng train. **Dự kiến là baseline mạnh nhất** vì 72% tháng đúng bằng 37,62 |
| F3 | Naive | Giá trị tháng cuối |
| F4 | Seasonal naive | Giá trị cùng tháng năm trước |
| F5 | ETS (Holt-Winters) | `statsmodels`, thử cấu hình có/không mùa vụ, damped trend; chọn theo validation |
| F6 | LightGBM trên lag | Dự báo đệ quy từng bước với feature §5.4; nhỏ (`num_leaves ≤ 7`) |

Mô hình mặc định cấp Tổng = mô hình có RMSE validation thấp nhất. Nếu khác biệt giữa các mô hình nằm trong sai số, chọn mô hình đơn giản hơn.

### 6.2 Album (cửa sổ 12 tháng)

| # | Mô hình | Cách tính |
|---|---|---|
| A1 | Global mean | Doanh thu trung bình một album trong cửa sổ train, giống nhau cho mọi album |
| A2 | Own-rate | `rev_rate` của chính album (doanh thu quy năm) |
| A3 | Size-proportional | `ŷ = k × n_tracks`, `k = Σy / Σn_tracks` trên train. **Baseline mạnh nhất dự kiến** |
| A4 | Poisson GLM | `PoissonRegressor` trên `log1p(feature)`, `alpha = 1` |
| A5 | **Hybrid LightGBM (Tweedie)** | 13 feature §5.4. **Mô hình mặc định** |

### 6.3 Track (cửa sổ 12 tháng)

| # | Mô hình | Cách tính |
|---|---|---|
| K1 | Zero | `ŷ = 0` (mốc tham chiếu cho các chỉ số, không dùng để dự báo) |
| K2 | Uniform allocation | `ŷ = ŷ_album / n_tracks` |
| K3 | Unsold allocation | Chia `ŷ_album` đều cho các track chưa từng bán, track đã bán nhận 0 |
| K4 | **Track LightGBM (Tweedie)** | 12 feature track §5.4. **Mô hình mặc định** |

K3 cố tình bỏ qua track đã bán, trong khi 29% doanh thu track 2013 đến từ track đã bán trước đó; K4 cần học được phần này.

### 6.4 Hòa giải phân cấp

| # | Phương pháp | Cách làm |
|---|---|---|
| R1 | Bottom-up | `album = Σ track`, `Tổng = Σ album` |
| R2 | **Top-down tỉ lệ (mặc định dự kiến)** | Giữ tổng 12 tháng từ mô hình Tổng tốt nhất; album = Tổng × `ŷ_album / Σ ŷ_album`; track = album × `ŷ_track / Σ ŷ_track` trong album. Hình dạng theo tháng lấy từ mô hình Tổng |
| R3 | Middle-out | Giữ `ŷ_album` (A5); track chia lại theo tỉ trọng K4 sao cho `Σ track = album`; `Tổng = Σ album` |

Lý do dự kiến chọn R2: cấp Tổng gần như hằng số nên tổng từ mô hình Tổng ít sai nhất, còn tổng các album dự báo thô lệch (thử nghiệm sơ bộ ở §7.2: ≈ 480–499 so với thực tế 450,6). Phương pháp mặc định cuối cùng chọn theo sai số 12 tháng của cấp Tổng và RMSE album trên validation.

---

## 7. Đánh giá

### 7.1 Cách đo

| Cấp | Chỉ số chính | Chỉ số phụ |
|---|---|---|
| Tổng | MAE, RMSE theo tháng | MAPE, MASE (thang = MAE naive trong train), sai số % của tổng 12 tháng; backtest 3 origin (2010-12, 2011-12 dùng cho chọn mô hình; 2012-12 là test) |
| Album | RMSE | MAE, WAPE (`Σ|e| / Σy`), độ lệch Tweedie, **bias** (`Σŷ / Σy − 1`), Spearman, NDCG@10/20 xếp theo doanh thu thực |
| Track | RMSE, độ lệch Poisson | WAPE tính trên nhóm (gộp theo album), bias, hit@K (K = 100, 500: số track trong top-K thực sự có bán) so với ngẫu nhiên (12,6%) |

- **Không dùng MAE làm chỉ số chính ở cấp Track** (xem §7.3).
- CI 95% bằng bootstrap theo album (1.000 lần; track lấy mẫu theo album chứa nó vì track cùng album tương quan). Cấp Tổng chỉ có 12 điểm test nên không bootstrap, thay bằng backtest nhiều origin.
- So sánh cặp với baseline mạnh nhất (A3 cho album) theo từng album: chênh lệch bình phương sai số trung bình và CI.
- **Khoảng dự báo 80%:** phân vị 10% và 90% của sai số trên validation (split conformal, tính riêng từng cấp; cận dưới cắt ở 0). Báo cáo độ phủ thực tế trên test.

### 7.2 Kết quả sơ bộ (khảo sát khả thi, chưa phải kết quả chính thức)

Đã chạy thử một lần trên dữ liệu thật để kiểm tra bài toán có tín hiệu hay không, dùng `HistGradientBoostingRegressor(loss='poisson')` của scikit-learn thay LightGBM. **Lần thử này có chạy cả trên cửa sổ test**, nên các số test bên dưới chỉ để đặt kỳ vọng. Kết quả chính thức sẽ chạy lại bằng pipeline đầy đủ, tham số chọn chỉ dựa trên validation.

**Album — RMSE (mô hình huấn luyện đến origin tương ứng)**

| Mô hình | RMSE validation (2012) | RMSE test (2013) |
|---|---|---|
| Dự báo 0 | 2,410 | 2,059 |
| A1 Global mean | 1,978 | 1,598 |
| A2 Own-rate | 1,941 | 1,600 |
| A3 Size-proportional | 1,723 | 1,292 |
| A4 Poisson GLM | 1,716 | 1,309 |
| A5 (bản thử, `num_leaves` 3 / 7) | 1,383 / 1,343 | 1,072 / 1,055 |

Tổng doanh thu album thực tế 2012 = 477,5 và 2013 = 450,6; tổng dự báo của A5 bản thử lần lượt 474–489 và 480–499, tức lệch dương khoảng 6–11% ở năm test. Đây là lý do cần bước hòa giải R2.

**Tổng — MAE tháng trên 2013 (bản thử, huấn luyện đến 2012-12)**

| Mô hình | MAE |
|---|---|
| F2 Median | 2,24 |
| F3 Naive | 2,24 |
| F1 Mean | 3,33 |
| F4 Seasonal naive | 4,41 |

Doanh thu thực tế 2013 = 450,58; Mean × 12 = 469,50 (lệch +4,2%). Median/Naive thắng vì doanh thu hằng số 37,62; seasonal naive thua vì sao chép luôn các đỉnh ngẫu nhiên của năm trước.

**Track (test 2013, album dùng A5 bản thử, tham chiếu cho K1–K3)**

| Mô hình | MAE | RMSE | Tổng dự báo (thực 450,6) |
|---|---|---|---|
| K1 Zero | 0,129 | 0,367 | 0 |
| K2 Uniform | 0,228 | 0,339 | 499,3 |
| K3 Unsold allocation | 0,206 | 0,351 | 490,8 |

Hit@100 của K3 = 33 và hit@500 = 132, so với ngẫu nhiên kỳ vọng ≈ 12,6 và ≈ 63. Spearman giữa dự báo và doanh thu thực, gộp theo album, = 0,875 (với K2).

### 7.3 Cách diễn giải dự kiến
- **Album:** có tín hiệu rõ so với mọi baseline, nhưng chưa có CI; tiêu chí đạt ở §7.4.
- **Tổng:** kỳ vọng F2/F3 thắng hoặc ngang các mô hình khác. Ghi nhận đúng như vậy.
- **Track:** K1 (toàn 0) có MAE tốt nhất vì 87% track-origin có nhãn 0, trong khi K2/K3 có RMSE tốt hơn K1 và xếp hạng tốt hơn ngẫu nhiên. Vì vậy dùng RMSE, độ lệch Poisson, WAPE theo nhóm và hit@K thay cho MAE. Mức độ chính xác ở cấp từng track thấp là kết luận chấp nhận được; giá trị chính nằm ở tổng theo album và xếp hạng.

### 7.4 Tiêu chí đạt
1. **Album:** A5 có RMSE test thấp hơn A3 (baseline mạnh nhất) và CI 95% của chênh lệch cặp không chứa 0. Nếu CI chứa 0, ghi nhận là chưa kết luận được, không tuyên bố vượt trội.
2. **Tổng:** mô hình được chọn không tệ hơn F2 về MAE test; nếu không có mô hình nào tốt hơn F2, mô hình mặc định là F2 và ghi rõ lý do.
3. **Track:** K4 hoặc K2 có RMSE thấp hơn K1 và hit@500 cao hơn ngẫu nhiên.
4. **Hòa giải:** NFR-07 đạt (sai lệch < 1e-6) và bias của tổng 12 tháng trên test giảm so với R1.
5. **Khoảng dự báo 80%:** độ phủ test báo cáo cùng giá trị danh định; chấp nhận độ phủ 65–95% do dữ liệu nhỏ.

### 7.5 Phân tích lỗi
Dự kiến thực hiện: sai số album theo thể loại, theo số track, theo việc album đã bán hay chưa bán trước origin; sai số tổng theo các tháng có dòng giá 1,99. Chưa thực hiện trong phiên bản này.

---

## 8. Cấu trúc dự án

```
BaitapChinook_Forecast/
├─ databases/
│  └─ Chinook_Sqlite.sqlite
├─ connectors/
│  └─ connector.py             # kết nối SQLite dùng chung (ensureConnection / close)
├─ models/                     # lớp dữ liệu thuần (dataclass) + model đã đóng gói
│  ├─ album.py                 # Album, Track
│  ├─ forecast.py              # Forecast (điểm, cận dưới, cận trên, lịch sử)
│  └─ forecaster.zip           # tạo bởi ml.train
├─ dal/                        # truy cập dữ liệu, nơi duy nhất viết SQL
│  ├─ RevenueDAO.py            # dòng doanh thu, chuỗi tháng, tổng theo album/track
│  └─ CatalogDAO.py            # catalog album, track, nghệ sĩ, thể loại
├─ ml/                         # học máy, không phụ thuộc Qt
│  ├─ data_prep.py             # làm sạch, dựng chuỗi, origin, kiểm tra rò rỉ
│  ├─ features.py              # FEATURE_COLUMNS, TRACK_FEATURE_COLUMNS, TOTAL_FEATURE_COLUMNS
│  ├─ forecasters.py           # F1–F6, A1–A5, K1–K4, huấn luyện LightGBM
│  ├─ reconcile.py             # R1–R3, kiểm tra nhất quán
│  ├─ evaluate.py              # metric, bootstrap CI, so sánh cặp, conformal
│  ├─ train.py                 # HUẤN LUYỆN
│  ├─ predict.py               # SUY LUẬN (Predictor), dòng lệnh
│  └─ package.py               # save_package / load_package
├─ ui/
│  ├─ ForecastMainWindow.ui    # thiết kế (Qt Designer)
│  ├─ ForecastMainWindow.py    # sinh bằng pyside6-uic, không sửa tay
│  ├─ ForecastMainWindowEx.py  # logic sự kiện, kế thừa Ui_MainWindow
│  ├─ widgets.py               # widget dùng chung (ô tìm kiếm có gợi ý)
│  └─ utils.py                 # showDataIntoTableWidget, drawForecastChart
├─ test/
│  └─ run_app.py               # mở ứng dụng (xem §8.2)
├─ artifacts/                  # metrics_total.csv, metrics_album.csv, metrics_track.csv, paired_vs_a3.csv, tuning, coherence.csv
├─ config.py                   # đường dẫn, seed, origin, horizon, lưới tham số
├─ main.py                     # chạy ứng dụng
└─ requirements.txt
```

### 8.1 Quy tắc lập trình

**A. Học máy**
1. Khám phá dữ liệu trước khi huấn luyện: đếm dòng, giá trị thiếu, đối chiếu doanh thu (ghi vào báo cáo làm sạch).
2. Feature khai báo tường minh (`FEATURE_COLUMNS`, `TRACK_FEATURE_COLUMNS`, `TOTAL_FEATURE_COLUMNS`). Cột văn bản được mã hóa trước khi đưa vào mô hình số.
3. **Chuỗi thời gian: không chia ngẫu nhiên, không xáo trộn.** Chia theo origin (§5.2); seed cố định.
4. Chỉ đánh giá trên test. Validation chỉ dùng để chọn tham số/mô hình/phương pháp hòa giải; nhãn validation không dùng để huấn luyện cửa sổ đang đánh giá.
5. Luôn so sánh với baseline đơn giản (F1–F4, A1–A3, K1–K2). Báo cáo chỉ số kèm CI hoặc backtest nhiều origin, không chỉ in dự báo.
6. Model lưu kèm tên feature. Khi suy luận kiểm tra tên và thứ tự feature khớp với code; sai thì báo lỗi rõ.
7. Huấn luyện (`ml/train.py`) và suy luận (`ml/predict.py`) tách riêng. Suy luận chỉ nạp model đã đóng gói.
8. Đường dẫn tính bằng `Path(__file__)` trong `config.py`.
9. Model đóng gói dạng `.zip` chứa `meta.json` (đọc được) và `model.pkl` (pickle). Chỉ nạp file do dự án tạo ra (NFR-06).
10. Dùng `logging`, không dùng `print`, cho tiến trình huấn luyện và kết quả.
11. Chuỗi giao diện và thông báo hoàn toàn tiếng Việt; tên biến, hàm, class tiếng Anh; comment tiếng Việt có dấu.
12. Dự báo luôn kẹp `max(0, ·)`; hòa giải chạy sau cùng và có `assert` nhất quán.

**B. Giao diện**
1. Tách thiết kế và logic: `.ui` (Qt Designer) → `.py` (sinh bằng `pyside6-uic`, không sửa tay) → `*Ex.py` (logic).
2. Thứ tự khởi tạo: `__init__` → `setupUi()` → `setupTables()` → `setupSignalAndSlot()` → `setupChart()` → `loadModel()` → `loadFilters()` → `showWindow()`.
3. Lỗi hiển thị bằng `QMessageBox` (critical cho lỗi hệ thống/model, warning cho nhập sai). Mọi handler bọc `try/except`.
4. Kiểm tra đầu vào trước khi xử lý (chưa chọn album/track).
5. Logic dữ liệu nằm ở DAO và `ml/`, UI chỉ gọi và hiển thị.
6. Bảng dùng chung `showDataIntoTableWidget(table, df)`.
7. SQL luôn dùng placeholder `?`.
8. Biểu đồ nhúng bằng `FigureCanvasQTAgg`; mỗi lần vẽ: `clear()` → vẽ → `tight_layout()` → `draw()`.
9. Hiển thị tiền với 2 chữ số thập phân và đơn vị; luôn kèm khoảng dự báo 80% bên cạnh điểm dự báo, vì doanh thu từng album/track rất nhiễu.

**C. Kiểm tra**
1. Thư mục `test/` chỉ chứa file khởi chạy giao diện, không có unit test.
2. Chống rò rỉ và nhất quán phân cấp được kiểm tra bằng `assert` trong `ml/data_prep.py` và `ml/reconcile.py`.

---

## 8.2 Thư mục `test/`

`test/run_app.py` mở ứng dụng để xem giao diện và thử dự báo, không cần chuẩn bị thêm. File có đoạn thêm thư mục gốc vào `sys.path`, nên chạy từ đâu cũng được.

Trước khi nộp, kiểm tra trong môi trường ảo mới: chỉ cài `requirements.txt`, chạy `python -m ml.train` để tạo model, rồi mở `test/run_app.py`; giao diện phải hiện và dự báo được cả ba cấp.

---

## 9. Giao diện

Một màn hình `ForecastMainWindow`, gồm:

| Khu vực | Nội dung |
|---|---|
| Thanh điều khiển | Ô chọn mô hình từng cấp (mặc định theo kết quả validation), ô chọn phương pháp hòa giải (mặc định R2), nút **Dự báo** |
| Tab **Tổng** | Đồ thị doanh thu tháng 2009–2013 và dự báo 12 tháng 2014 kèm dải 80%; thẻ tóm tắt: tổng 12 tháng dự báo, so với 12 tháng gần nhất |
| Tab **Album** | Ô tìm album (theo tên hoặc nghệ sĩ), bộ lọc thể loại. Bảng: #, album, nghệ sĩ, thể loại, doanh thu 12 tháng gần nhất, dự báo 12 tháng, khoảng 80%, thay đổi % |
| Tab **Track** | Chọn album, bảng track của album: track, thể loại, giá, đã bán trước đó (có/không), dự báo 12 tháng, khoảng 80% |
| Đồ thị chi tiết | Lịch sử doanh thu theo năm và dự báo của đối tượng đang chọn (tổng, album hoặc track) |
| Hộp **Độ tin cậy** | Hiển thị RMSE/WAPE test của mô hình đang chọn ở cấp tương ứng và ghi chú "dự báo từng track rất nhiễu" |
| Thanh trạng thái | Phương pháp hòa giải đang dùng, kiểm tra nhất quán (`Σ track = album = tổng`) |

**Luồng dự báo**
- Nạp `models/forecaster.zip` → kiểm tra tên feature → dự báo cấp Tổng (12 tháng) → dự báo album bằng mô hình đã chọn → dự báo track → hòa giải → khoảng 80% → hiển thị.
- Khi chọn một album, tab Track hiển thị track của album đó với dự báo đã hòa giải.

---

## 10. Kế hoạch & sản phẩm bàn giao

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 0 | Khảo sát dữ liệu, thử baseline, thiết kế (tài liệu này) | Xong |
| 1 | `ml/data_prep.py`: làm sạch, chuỗi tháng, origin, kiểm tra rò rỉ | Chưa làm |
| 2 | `ml/evaluate.py`, baseline F1–F4, A1–A3, K1–K2 | Chưa làm |
| 3 | F5–F6, A4–A5, K3–K4, tuning, hòa giải, `ml/train.py`, `ml/predict.py`, đóng gói | Chưa làm |
| 4 | Khoảng dự báo (conformal), CI, so sánh cặp, phân tích lỗi | Chưa làm |
| 5 | Giao diện một màn hình, ba tab, đồ thị | Chưa làm |
| 6 | README, cập nhật TECHSPEC bằng kết quả chính thức, báo cáo ngắn | Chưa làm |

**Bàn giao:** mã nguồn, `databases/Chinook_Sqlite.sqlite`, `models/forecaster.zip`, `requirements.txt`, file `.ui` và `.py` sinh ra, các CSV trong `artifacts/`, `README.md`, tài liệu này.

---

## 11. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Doanh thu Tổng gần hằng số, mô hình học máy không thắng baseline | Giữ F2/F3 làm baseline bắt buộc; ghi nhận kết luận đúng như đo được; không ép mô hình phức tạp |
| Dữ liệu nhỏ (60 tháng, 59 customer) → kết quả dao động | Backtest nhiều origin, bootstrap theo album, so sánh cặp, báo cáo CI |
| Chuỗi tháng cấp album/track quá thưa | Dự báo theo cửa sổ 12 tháng, không theo tháng |
| Nhãn train chồng lấn (origin cách 6 tháng, nhãn 12 tháng) làm CI quá hẹp | Ghi nhận; chọn tham số bằng validation tách riêng; không dùng CI như bằng chứng tuyệt đối |
| Rò rỉ dữ liệu | Feature chỉ dùng ≤ origin; `assert` về ngày; nhãn train kết thúc trước origin đánh giá |
| Dùng MAE ở cấp Track cho kết quả sai lệch | Dùng RMSE, độ lệch Poisson, WAPE theo nhóm, hit@K; nêu rõ trong §7 |
| Tổng ba cấp không khớp nhau | Hòa giải R1–R3, `assert` nhất quán (NFR-07) |
| Catalog không có ngày phát hành, album mới không dự báo được | Ghi nhận hạn chế; ngoài phạm vi |
| Tháng 12/2013 chưa đủ tháng | Ghi chú trong báo cáo làm sạch; doanh thu tháng này vẫn ở mức bình thường |
| Khoảng dự báo dựa trên ít điểm validation | Báo cáo độ phủ thực tế trên test; chấp nhận 65–95% |
| Pickle có thể chạy code tùy ý | Chỉ nạp file do dự án tạo ra (NFR-06) |
| Model cũ không khớp code | Kiểm tra tên feature khi nạp; báo lỗi và hướng dẫn chạy lại `ml.train` |
| Dữ liệu mới không tự vào model | Phải đưa dữ liệu vào DB và huấn luyện lại |

---

## 12. Các quyết định đã chốt

- Ba cấp dự báo: **Tổng (theo tháng), Album, Track**, horizon 12 tháng, doanh thu = `UnitPrice × Quantity`.
- Album/Track dự báo theo **cửa sổ 12 tháng** tại từng origin (chuỗi tháng quá thưa); Tổng dự báo theo tháng.
- Chia theo **origin thời gian**: test origin 2012-12-31 (nhãn 2013), validation origin 2011-12-31 (nhãn 2012); không chia ngẫu nhiên.
- Track dự báo bằng phân bổ từ album và mô hình Tweedie với feature của album; không dựng chuỗi thời gian riêng cho từng track.
- Mô hình: classical ML (baseline, Poisson GLM, ETS) và LightGBM Tweedie; không deep learning.
- Hòa giải phân cấp bắt buộc; phương pháp mặc định dự kiến R2 (top-down tỉ lệ), xác nhận bằng validation.
- Cấp Tổng: mô hình mặc định là mô hình thắng trên validation; dự kiến F2 (Median). Ghi nhận trung thực nếu mô hình học máy không vượt baseline.
- Chỉ số: RMSE, WAPE, bias, độ lệch Tweedie/Poisson, hit@K; không dùng MAE làm chỉ số chính cho Track.
- Huấn luyện offline bằng `python -m ml.train`. Ứng dụng chỉ suy luận từ `models/forecaster.zip`.
- UI: Qt Designer + PySide6, tiếng Việt, một màn hình ba tab, cấu trúc `ui / dal / models / ml / test`.
- Không đóng gói `.exe`.

**Giả định cần xác nhận:** horizon 12 tháng và đơn vị "doanh thu" tính theo dòng hóa đơn (không trừ chi phí/thuế); nếu muốn horizon 6 tháng hoặc dự báo theo quý thì chỉ đổi `HORIZON_MONTHS` và danh sách origin trong `config.py`.
