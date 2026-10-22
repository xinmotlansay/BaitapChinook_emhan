"""Logic của cửa sổ chính. Không sửa MainWindow.py (do pyuic6 sinh ra từ MainWindow.ui)."""
import html
import json

import numpy as np
import pandas as pd
from matplotlib import colormaps
from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QMainWindow

from config import ARTIFACTS_DIR
from dal.CustomerDAO import CustomerDAO
from dal.InteractionDAO import InteractionDAO
from ml.data_prep import mark_item_cold, split_by_time
from ml.predict import explain, list_models, model_meta, recommend
from ui.MainWindow import Ui_MainWindow
from ui.utils import (GROUP_LABELS, LEVEL_LABELS, METRIC_LABELS, SIGNAL_LABELS, Worker, fmt_ci, fmt_date,
                      model_label, show_error)
from ui.widgets import SortableTable, attach_canvas

ACCENT = "#d95f02"  # mô hình chính trên biểu đồ so sánh
BAR_COLOR = "#4c78a8"


class MainWindowEx(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pool = QThreadPool.globalInstance()
        self.customers = {}
        self.metrics = None
        self._recs, self._ctx = [], None  # kết quả gợi ý đang hiển thị và (khách, kind, mô hình) sinh ra nó

        self.history_table = SortableTable(self.tblHistory)
        self.result_table = SortableTable(self.tblResult)
        self.metrics_table = SortableTable(self.tblMetrics, stretch_last=False)
        self.genre_canvas = attach_canvas(self.chartHost)
        self.compare_canvas = attach_canvas(self.chartHostCompare)
        self.splitTables.setSizes([360, 470, 420])

        self._fill_combos()
        self._connect()
        self._load_customers()
        self._load_metrics()
        self._load_info()

    # ---------------------------------------------------------------- khởi tạo
    def _fill_combos(self):
        for model_id in list_models():
            self.cboModel.addItem(model_label(model_id), model_id)
        for key, label in LEVEL_LABELS.items():
            self.cboLevel.addItem(label, key)
        for key, label in GROUP_LABELS.items():
            self.cboGroup.addItem(label, key)
        for key, label in METRIC_LABELS.items():
            self.cboMetric.addItem(label, key)
        self.cboMetric.setCurrentIndex(list(METRIC_LABELS).index("ndcg"))

    def _connect(self):
        self.cboCustomer.currentIndexChanged.connect(self.on_customer_changed)
        self.btnRecommend.clicked.connect(self.on_recommend)
        self.tblResult.selectionModel().currentRowChanged.connect(self.on_result_selected)
        for combo in (self.cboLevel, self.cboGroup, self.cboMetric):
            combo.currentIndexChanged.connect(self.refresh_compare)

    # ---------------------------------------------------------------- tab 1
    def _load_customers(self):
        try:
            customers = CustomerDAO.get_all()
        except Exception as e:
            show_error(self, f"Không đọc được danh sách khách hàng:\n{e}")
            return
        self.customers = {c.customer_id: c for c in customers}
        self.cboCustomer.blockSignals(True)
        for c in customers:
            self.cboCustomer.addItem(f"{c.customer_id} — {c.name}", c.customer_id)
        self.cboCustomer.blockSignals(False)
        if customers:
            self.on_customer_changed()

    def current_customer_id(self):
        return self.cboCustomer.currentData()

    def on_customer_changed(self, _=None):
        cid = self.current_customer_id()
        if cid is None:
            return
        c = self.customers[cid]
        hist = InteractionDAO.get_history(cid)
        self.lblCountry.setText(f"Quốc gia: {c.country or '—'}")
        self.lblBought.setText(f"Đã mua: {len(hist)} track")

        shown = pd.DataFrame({
            "Track": hist.track, "Album": hist.album, "Nghệ sĩ": hist.artist,
            "Thể loại": hist.genre, "Ngày": hist.invoice_date.map(fmt_date),
        })
        raw = shown.assign(**{"Ngày": hist.invoice_date})
        self.history_table.set_data(shown, raw)
        self.result_table.set_data(pd.DataFrame())
        self._recs, self._ctx = [], None
        self.txtDetail.clear()
        self.draw_genre_chart(hist)

    def draw_genre_chart(self, hist):
        if hist.empty:
            self.genre_canvas.message("Khách chưa mua track nào")
            return
        counts = hist.genre.value_counts().head(10).sort_values()
        ax = self.genre_canvas.reset()
        bars = ax.barh(counts.index, counts.values, color=BAR_COLOR)
        ax.bar_label(bars, padding=3, color=ax.xaxis.label.get_color())
        ax.set_xlabel("Số track đã mua")
        ax.margins(x=0.12)
        self.genre_canvas.draw_idle()

    def on_recommend(self):
        cid = self.current_customer_id()
        if cid is None:
            return
        kind = "track" if self.rdoTrack.isChecked() else "album"
        k, model = self.spinK.value(), self.cboModel.currentData()
        self.btnRecommend.setEnabled(False)
        self.btnRecommend.setText("Đang tính...")

        worker = Worker(lambda: recommend(cid, kind, k, model))
        worker.signals.finished.connect(lambda recs: self.on_recommend_done(recs, kind, cid, model))
        worker.signals.failed.connect(self.on_recommend_failed)
        self.pool.start(worker)

    def on_recommend_done(self, recs, kind, cid, model):
        self.btnRecommend.setEnabled(True)
        self.btnRecommend.setText("Gợi ý")
        label = "Track" if kind == "track" else "Album"
        df = pd.DataFrame({
            "#": range(1, len(recs) + 1), label: [r.name for r in recs],
            "Nghệ sĩ": [r.artist_name for r in recs], "Điểm": [float(r.score) for r in recs],
            "Lý do": [r.reason for r in recs],
        })
        self.result_table.set_data(df)
        self.grpResult.setTitle(f"Kết quả gợi ý — {label}")
        self._recs, self._ctx = recs, (cid, kind, model)
        self.txtDetail.clear()

    def on_result_selected(self, current, _previous):
        """Bấm vào một dòng gợi ý -> hiện chi tiết lý do trong khung bên dưới bảng."""
        if not current.isValid() or not self._recs:
            return
        rec = self._recs[self.result_table.proxy.mapToSource(current).row()]
        cid, kind, model = self._ctx
        try:
            self.txtDetail.setHtml(self._detail_html(rec, explain(cid, rec.item_id, kind, model), kind))
        except Exception as e:
            self.txtDetail.setPlainText(f"Không lấy được chi tiết: {e}")

    @staticmethod
    def _detail_html(rec, info, kind):
        e = html.escape
        total = info["bought_total"]
        pct = f" ({info['same_genre'] / total:.0%})" if total else ""
        parts = [
            f"<h3>{e(rec.name)}</h3>",
            f"<p><b>Lý do:</b> {e(rec.reason)}</p>",
            f"<p>Nghệ sĩ: {e(info['artist'])} · Album: {e(info['album'])} · Thể loại: {e(info['genre'])}"
            + (f" · {info['n_tracks']} track" if kind == "album" else "") + f" · Điểm: {rec.score:.4f}</p>",
            f"<p><b>Đối chiếu với lịch sử mua của bạn ({total} track):</b></p><ul>",
            f"<li>{info['same_artist']} track cùng nghệ sĩ {e(info['artist'])}</li>",
            f"<li>{info['same_genre']} track cùng thể loại {e(info['genre'])}{pct}</li>",
            f"<li>{info['same_album']} track cùng album</li></ul>",
        ]
        if info["signals"]:
            rows = []
            for key, (label, style) in SIGNAL_LABELS.items():
                v = info["signals"][key]
                text = {"int": f"{v:.0f}", "pct": f"{v:.1%}", "num": f"{v:.4f}"}[style]
                rows.append(f"<tr><td>{label}</td><td align='right'>&nbsp;&nbsp;{text}</td></tr>")
            parts.append("<p><b>Tín hiệu của mô hình Hybrid:</b></p><table>" + "".join(rows) + "</table>")
        return "".join(parts)

    def on_recommend_failed(self, message):
        self.btnRecommend.setEnabled(True)
        self.btnRecommend.setText("Gợi ý")
        show_error(self, message)

    # ---------------------------------------------------------------- tab 2
    def _load_metrics(self):
        path = ARTIFACTS_DIR / "metrics.csv"
        if not path.exists():
            self.metrics = None
            self.compare_canvas.message("Chưa có artifacts/metrics.csv — chạy: python -m ml.train")
            return
        self.metrics = pd.read_csv(path)
        self.refresh_compare()

    def refresh_compare(self, _=None):
        if self.metrics is None:
            return
        level, group, metric = self.cboLevel.currentData(), self.cboGroup.currentData(), self.cboMetric.currentData()
        sel = self.metrics[(self.metrics.level == level) & (self.metrics.group == group)
                           & (self.metrics.metric == metric)]
        if sel.empty:
            self.metrics_table.set_data(pd.DataFrame())
            self.compare_canvas.message("Không có dữ liệu cho lựa chọn này")
            return

        order = [m for m in list_models() if m in set(sel.model)]
        ks = sorted(sel.k.unique())
        text, raw = [], []
        for m in order:
            rows = sel[sel.model == m].set_index("k")
            text.append([model_label(m)] + [fmt_ci(*rows.loc[k, ["mean", "ci_lo", "ci_hi"]]) for k in ks])
            raw.append([model_label(m)] + [float(rows.loc[k, "mean"]) for k in ks])
        cols = ["Mô hình"] + [f"K={k}" for k in ks]
        self.metrics_table.set_data(pd.DataFrame(text, columns=cols), pd.DataFrame(raw, columns=cols))
        self.metrics_table.view.setColumnWidth(0, 260)
        self.draw_compare_chart(sel, order, ks, level, metric)

    def draw_compare_chart(self, sel, order, ks, level, metric):
        ax = self.compare_canvas.reset()
        n = len(order)
        width = 0.8 / n
        shades = colormaps["Blues"](np.linspace(0.3, 0.7, max(n - 1, 1)))
        baseline_i = 0
        for i, m in enumerate(order):
            rows = sel[sel.model == m].set_index("k").loc[ks]
            mean = rows["mean"].to_numpy(dtype=float)
            err = np.vstack([np.nan_to_num(mean - rows.ci_lo.to_numpy(dtype=float)),
                             np.nan_to_num(rows.ci_hi.to_numpy(dtype=float) - mean)]).clip(min=0)
            if m == "M_hybrid":
                color = ACCENT
            else:
                color, baseline_i = shades[baseline_i], baseline_i + 1
            x = np.arange(len(ks)) + (i - (n - 1) / 2) * width
            ax.bar(x, mean, width * 0.92, yerr=err, color=color, label=model_label(m),
                   error_kw=dict(ecolor=ax.xaxis.label.get_color(), elinewidth=0.8, capsize=2))
        ax.set_xticks(np.arange(len(ks)), [f"K={k}" for k in ks])
        ax.set_ylabel(f"{METRIC_LABELS[metric]} ({LEVEL_LABELS[level]})")
        ax.legend(fontsize=7, frameon=False, ncols=2, labelcolor=ax.xaxis.label.get_color())
        self.compare_canvas.draw_idle()

    # ---------------------------------------------------------------- tab 3
    def _load_info(self):
        try:
            self.txtDataInfo.setPlainText(self._data_summary())
        except Exception as e:
            self.txtDataInfo.setPlainText(f"Không tính được thống kê dữ liệu: {e}")
        try:
            self.txtModelInfo.setPlainText(self._model_summary(model_meta()))
        except FileNotFoundError:
            self.txtModelInfo.setPlainText("Chưa có gói mô hình. Chạy: python -m ml.train")
        except Exception as e:
            self.txtModelInfo.setPlainText(f"Không đọc được gói mô hình: {e}")

    @staticmethod
    def _data_summary():
        split = split_by_time()
        inter = InteractionDAO.get_all()
        n_cust, n_tracks = inter.customer_id.nunique(), len(split.tracks)
        buyers = inter.groupby("track_id").customer_id.nunique()
        bought_albums = split.tracks.loc[buyers.index, "album_id"].nunique()
        cold = mark_item_cold(split.test, split.train_plus_val()).item_cold.mean()
        return "\n".join([
            f"Khách hàng: {n_cust}",
            f"Track trong catalog: {n_tracks}",
            f"Tương tác (khách, track): {len(inter)}",
            f"Độ đặc ma trận khách × track: {len(inter) / (n_cust * n_tracks):.2%}",
            f"Track được mua: {len(buyers)}; được ≥2 khách mua: {(buyers >= 2).sum()}",
            f"Album có track được mua: {bought_albums}",
            f"Thời gian: {inter.invoice_date.min():%d/%m/%Y} → {inter.invoice_date.max():%d/%m/%Y}",
            f"Hóa đơn / khách (trung bình): {inter.groupby('customer_id').invoice_id.nunique().mean():.1f}",
            "",
            "Chia theo thời gian (hóa đơn cuối = test, kế cuối = val):",
            f"  train: {len(split.train)} dòng",
            f"  val:   {len(split.val)} dòng",
            f"  test:  {len(split.test)} dòng",
            f"Tỷ lệ dòng test là item-cold (chưa ai mua ở train+val): {cold:.1%}",
        ])

    @staticmethod
    def _model_summary(meta):
        lines = [
            f"Mô hình: {meta['model_name']}",
            f"Huấn luyện lúc: {meta['trained_at']}",
            f"Seed: {meta['seed']}",
            f"Mẫu huấn luyện: {meta['train_info']['n_pos']} dương, {meta['train_info']['n_neg']} âm "
            f"({meta['train_info']['n_folds']} fold trong)",
            f"Gộp điểm album: {' + '.join(meta['album_agg'])}",
            f"GBDT: {json.dumps(meta['gbdt_params'])}",
            "",
            "Thư viện: " + ", ".join(f"{k} {v}" for k, v in meta["libraries"].items()),
            "",
            f"Feature ({len(meta['feature_columns'])}): " + ", ".join(meta["feature_columns"]),
            "",
            "Metric trên test (Top-10, mean [CI 95 %]):",
        ]
        for name, v in sorted(meta.get("metrics", {}).items()):
            lines.append(f"  {name}: {fmt_ci(v['mean'], v['ci_lo'], v['ci_hi'])}")
        return "\n".join(lines)
