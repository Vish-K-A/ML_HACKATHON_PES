"""Assemble the technical report PDF.

report/report.md contains the written text with placeholders of the form
{{table:NAME}} and {{value:NAME}}. Every table and number is filled in here
directly from the CSV files in results/, so the PDF always matches the
experiment outputs. The Markdown is rendered to HTML and printed to PDF with
headless Chrome/Edge.

Usage: python scripts/06_build_report.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import markdown
import pandas as pd

from pathsnap.config import RESULTS_DIR, ROOT

REPORT = ROOT / "report"
R = RESULTS_DIR


def f(x, d=4):
    return "N/A" if pd.isna(x) else f"{x:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def tables() -> dict[str, str]:
    T = {}
    inv = pd.read_csv(R / "eda" / "inventory.csv", dtype={"trip": str})
    tot = inv.groupby("partition").agg(trips=("trip", "count"), gps_points=("n_gps", "sum"),
                                        median_points=("n_gps", "median"), route_arcs=("n_route", "sum"),
                                        median_arcs_network=("n_arcs", "median"),
                                        gaps_over_30s=("n_gaps_gt30s", "sum"),
                                        mean_cand_miss_pct=("candidate_miss_pct", "mean")).reset_index()
    tot["mean_cand_miss_pct"] = tot.mean_cand_miss_pct.map(lambda v: f"{v:.2f}")
    T["inventory"] = md_table(tot)
    worst = inv.sort_values("pct_gps_gt60m_from_route", ascending=False).head(6)
    w = worst[["partition", "trip", "n_gps", "track_km", "route_km", "align_dist_median_m", "pct_gps_gt60m_from_route",
               "candidate_miss_pct"]].copy()
    for c in ["track_km", "route_km", "align_dist_median_m", "pct_gps_gt60m_from_route", "candidate_miss_pct"]:
        w[c] = w[c].map(lambda v: f"{v:.1f}")
    T["anomalies"] = md_table(w)
    miss = pd.read_csv(R / "eda" / "missing_values.csv")
    T["missing"] = md_table(miss.rename(columns={miss.columns[0]: "item"}))
    sens = pd.read_csv(R / "eda" / "candidate_sensitivity.csv")
    piv = sens.pivot(index="radius_m", columns="k", values="miss_pct").round(2).reset_index()
    piv.columns = ["radius (m) \\ K"] + [f"K={c}" for c in piv.columns[1:]]
    T["cand_sens"] = md_table(piv)

    mc = pd.read_csv(R / "models" / "model_comparison.csv").sort_values("test_top1_accuracy", ascending=False)
    t = pd.DataFrame({
        "Model": mc.model, "Family": mc.family, "Key configuration": mc.config,
        "CV macro F1": [f"{a:.4f} ± {b:.4f}" for a, b in zip(mc.cv_macro_f1_mean, mc.cv_macro_f1_std)],
        "Top-1 acc. (test)": mc.test_top1_accuracy.map(f),
        "Route continuity (test)": mc.test_route_continuity.map(f),
        "Test macro F1": mc.test_macro_f1.map(f), "Test log loss": mc.test_log_loss.map(f),
        "Inference (ms / 1k rows)": mc.inference_ms_per_1k_rows.map(lambda v: f"{v:.2f}")})
    T["model_comparison"] = md_table(t)
    t2 = pd.DataFrame({"Model": mc.model, "Accuracy": mc.test_accuracy.map(f), "Precision": mc.test_precision.map(f),
                       "Recall": mc.test_recall.map(f), "F1 (pos.)": mc.test_f1.map(f),
                       "CV Top-1": [f"{a:.4f} ± {b:.4f}" for a, b in zip(mc.cv_top1_mean, mc.cv_top1_std)],
                       "Train macro F1": mc.train_macro_f1.map(f), "CV log loss": mc.cv_log_loss_mean.map(f),
                       "Fit time (s)": mc.fit_time_s.map(lambda v: f"{v:.1f}")})
    T["model_details"] = md_table(t2)
    for _, r in mc.iterrows():
        cm = json.loads(r.test_confusion)
        T[f"m_{r.model}"] = md_table(pd.DataFrame([
            {"Metric": "Observation-level Top-1 accuracy (student test)", "Value": f(r.test_top1_accuracy)},
            {"Metric": "Macro F1 (student test, candidate level)", "Value": f(r.test_macro_f1)},
            {"Metric": "Log loss (student test)", "Value": f(r.test_log_loss)},
            {"Metric": "Precision / Recall / F1 of positive class", "Value": f"{f(r.test_precision)} / {f(r.test_recall)} / {f(r.test_f1)}"},
            {"Metric": "Route continuity, classifier only", "Value": f(r.test_route_continuity)},
            {"Metric": "CV macro F1 (5-fold, mean ± sd)", "Value": f"{f(r.cv_macro_f1_mean)} ± {f(r.cv_macro_f1_std)}"},
            {"Metric": "CV Top-1 (mean ± sd)", "Value": f"{f(r.cv_top1_mean)} ± {f(r.cv_top1_std)}"},
            {"Metric": "Training-set macro F1 (refit)", "Value": f(r.train_macro_f1)},
            {"Metric": "Confusion matrix [[TN, FP], [FN, TP]]", "Value": str(cm)},
            {"Metric": "Training rows / fit time", "Value": f"{r.train_rows:,} / {r.fit_time_s:.1f} s"},
        ]))
    folds = pd.read_csv(R / "models" / "cv_folds.csv")
    fp = folds.pivot(index="model", columns="fold", values="top1_accuracy").round(4).reset_index()
    fp.columns = ["Model"] + [f"fold {c}" for c in fp.columns[1:]]
    T["cv_folds_top1"] = md_table(fp)

    import glob
    for p in glob.glob(str(R / "optuna" / "tuned_vs_untuned_*.csv")):
        c = pd.read_csv(p, index_col=0)
        rows = []
        for k, lab in [("cv_macro_f1", "CV macro F1 (5-fold)"), ("cv_top1", "CV Top-1"), ("cv_log_loss", "CV log loss"),
                       ("test_top1_accuracy", "Student-test Top-1"), ("test_macro_f1", "Student-test macro F1"),
                       ("test_log_loss", "Student-test log loss"), ("test_route_continuity", "Student-test continuity"),
                       ("fit_time_s", "Fit time (s)"), ("inference_time_s", "Inference time, 160k rows (s)")]:
            rows.append({"Metric": lab, "Untuned": f(float(c.loc["untuned", k])), "Tuned": f(float(c.loc["tuned", k]))})
        T["tuned_vs_untuned"] = md_table(pd.DataFrame(rows))
        best = json.loads(c.loc["tuned", "params"])
        T["best_params"] = md_table(pd.DataFrame([{"Hyperparameter": k, "Best value": (f"{v:.5g}" if isinstance(v, float) else v)}
                                                  for k, v in best.items()]))
        T["n_trials"] = str(int(float(c.loc["tuned", "n_trials"])))
        T["search_time"] = f"{float(c.loc['tuned', 'search_time_s']) / 60:.1f}"
    tr = sorted(glob.glob(str(R / "optuna" / "trials_*.csv")))
    if tr:
        td = pd.read_csv(tr[0]).sort_values("value", ascending=False).head(8)
        keep = ["number", "value"] + [c for c in td.columns if c.startswith("params_")] + ["user_attrs_macro_f1", "user_attrs_log_loss"]
        td = td[keep].copy()
        td.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in td.columns]
        for c in td.columns[1:]:
            td[c] = td[c].map(lambda v: f"{v:.4g}" if isinstance(v, float) else v)
        T["top_trials"] = md_table(td.rename(columns={"value": "CV Top-1"}))

    if (R / "hmm" / "student_test_classifier_vs_hmm.csv").exists():
        h = pd.read_csv(R / "hmm" / "student_test_classifier_vs_hmm.csv", dtype={"trip": str})
        ht = pd.DataFrame({"Trip": h.trip, "Points": h.n_points.astype(int),
                           "Classifier acc.": h.clf_acc.map(f), "HMM acc.": h.hmm_acc.map(f),
                           "Clf continuity": h.clf_continuity.map(f), "HMM continuity": h.hmm_continuity.map(f),
                           "Clf route F1": h.clf_route_f1.map(f), "HMM route F1": h.hmm_route_f1.map(f),
                           "HMM route P / R": [f"{f(a, 3)} / {f(b, 3)}" for a, b in zip(h.hmm_route_precision, h.hmm_route_recall)],
                           "Breaks": h.n_breaks.astype(int)})
        T["hmm_test"] = md_table(ht)
        g = pd.read_csv(R / "hmm" / "decoder_grid_summary.csv")
        g = g.sort_values("hmm_acc", ascending=False)
        for c in ["hmm_acc", "clf_acc", "hmm_route_f1", "hmm_continuity"]:
            g[c] = g[c].map(f)
        T["decoder_grid"] = md_table(g)
        gr = pd.read_csv(R / "hmm" / "gap_robustness.csv")
        gr["points"] = gr.points.astype(int)
        for c in gr.columns[2:]:
            gr[c] = gr[c].map(f)
        T["gap"] = md_table(gr)
        cfg = json.loads((ROOT / "models" / "pathsnap_final_config.json").read_text())
        T["hmm_params"] = md_table(pd.DataFrame([{"Parameter": k, "Value": v} for k, v in cfg["hmm_params"].items()]))
    return T


def values() -> dict[str, str]:
    V = {}
    mc = pd.read_csv(R / "models" / "model_comparison.csv").set_index("model")
    for m, r in mc.iterrows():
        for k in ["test_top1_accuracy", "test_macro_f1", "test_log_loss", "cv_macro_f1_mean", "test_route_continuity",
                  "cv_top1_mean", "train_macro_f1", "fit_time_s", "inference_ms_per_1k_rows", "cv_top1_std"]:
            V[f"{m}.{k}"] = f(r[k], 4 if "time" not in k and "ms" not in k else 2)
            if k in ("test_top1_accuracy", "cv_top1_mean", "test_route_continuity"):
                V[f"{m}.{k}.pct"] = f"{r[k] * 100:.2f}%"
    if (R / "hmm" / "student_test_classifier_vs_hmm.csv").exists():
        h = pd.read_csv(R / "hmm" / "student_test_classifier_vs_hmm.csv", dtype={"trip": str}).iloc[-1]
        for k in h.index[1:]:
            V[f"hmm.{k}"] = f(float(h[k]))
            V[f"hmm.{k}.pct"] = f"{float(h[k]) * 100:.2f}%"
    inv = pd.read_csv(R / "eda" / "inventory.csv", dtype={"trip": str})
    V["eda.utm_zones"] = str(inv.utm_epsg.nunique())
    V["eda.max_gap_h"] = f"{inv.dt_max.max() / 3600:.1f}"
    V["eda.train_points"] = f"{inv[inv.partition == 'train'].n_gps.sum():,}"
    V["eda.test_points"] = f"{inv[inv.partition == 'student_test'].n_gps.sum():,}"
    V["eda.align_median"] = f"{inv.align_dist_median_m.median():.1f}"
    V["eda.pct_1s"] = f"{inv.pct_dt_1s.mean():.1f}"
    V["eda.two_way"] = f"{inv.pct_two_way_arcs.mean():.0f}"
    pts = pd.read_parquet(R / "points_train.parquet")
    V["train.miss_pct"] = f"{(1 - pts.cand_hit.mean()) * 100:.2f}"
    pte = pd.read_parquet(R / "points_student_test.parquet")
    V["test.miss_pct"] = f"{(1 - pte.cand_hit.mean()) * 100:.2f}"
    ft = pd.read_parquet(R / "features_train.parquet", columns=["label"])
    V["train.rows"] = f"{len(ft):,}"
    V["train.pos_rate"] = f"{ft.label.mean() * 100:.1f}"
    V["train.rows_filtered"] = f"{int(pd.read_csv(R / 'models' / 'model_comparison.csv').train_rows.max()):,}"
    return V


def find_browser() -> str | None:
    for p in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]:
        if Path(p).exists():
            return p
    return shutil.which("chromium") or shutil.which("google-chrome")


CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1b1f24; }
h1 { font-size: 21pt; margin-bottom: 0; } h2 { font-size: 15pt; border-bottom: 2px solid #0f766e; padding-bottom: 3px; margin-top: 26px; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 18px; page-break-after: avoid; color: #0f4c45 } h4 { font-size: 10.5pt; margin: 10px 0 4px; }
table { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 8px 0 12px; page-break-inside: avoid; }
th, td { border: 1px solid #c9d1d9; padding: 3px 5px; vertical-align: top; } th { background: #e8f3f1; text-align: left; }
tr:nth-child(even) td { background: #f8fafb; }
img { max-width: 100%; display: block; margin: 8px auto; page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f2f4f6; padding: 0 3px; }
pre { background: #f2f4f6; padding: 8px; font-size: 8.5pt; white-space: pre-wrap; }
blockquote { border-left: 4px solid #0f766e; margin: 8px 0; padding: 4px 12px; background: #f1f8f7; }
.cap { text-align: center; font-size: 8.8pt; color: #555; margin-top: -4px; }
"""


def main():
    src = (REPORT / "report.md").read_text(encoding="utf-8")
    T, V = tables(), values()

    def rep_t(m):
        return T.get(m.group(1), f"**[missing table {m.group(1)}]**")

    def rep_v(m):
        return V.get(m.group(1), f"**[missing value {m.group(1)}]**")

    src = re.sub(r"\{\{table:([\w\-.]+)\}\}", rep_t, src)
    src = re.sub(r"\{\{value:([\w\-.%]+)\}\}", rep_v, src)
    missing = re.findall(r"\[missing (?:table|value) [^\]]+\]", src)
    if missing:
        print("WARNING missing placeholders:", missing)
    (REPORT / "report_filled.md").write_text(src, encoding="utf-8")
    body = markdown.markdown(src, extensions=["tables", "fenced_code", "toc"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap Technical Report</title><style>{CSS}</style></head><body>{body}</body></html>"
    out_html = REPORT / "report.html"
    out_html.write_text(html, encoding="utf-8")
    br = find_browser()
    if br:
        pdf = REPORT / "PathSnap_Technical_Report.pdf"
        subprocess.run([br, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", out_html.resolve().as_uri()], check=True, timeout=180)
        print("PDF written:", pdf)
    else:
        print("No Chrome/Edge found; open report/report.html and print to PDF.")


if __name__ == "__main__":
    main()
