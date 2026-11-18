"""Train and compare candidate classifiers from several algorithm families.

Protocol
  * learning task: (observation, candidate) -> is reference arc?  (binary)
  * training rows: train partition, observations whose reference arc is in the
    candidate set (others carry no positive and mostly reflect incomplete
    reference routes, see data_quality_notes.md)
  * 5-fold GroupKFold by trip on the 80 training trips (no trip in two folds)
  * development comparison on the 10 student_test trips (all observations;
    candidate misses count as Top-1 errors)
  * same features, preprocessing and seed for every model

Outputs (results/models/): model_comparison.csv, cv_folds.csv, experiment_log.jsonl,
confusion matrices, comparison plots, MLP loss curve, per-model test scores.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pathsnap.config import RESULTS_DIR, SEED
from pathsnap.features import FEATURES
from pathsnap.io import load_trip
from pathsnap.metrics import classification_metrics, continuity_from_arcs, top1_predictions

OUT = RESULTS_DIR / "models"
OUT.mkdir(parents=True, exist_ok=True)


def model_zoo():
    """name -> (family, estimator factory, config description, max training points)."""
    return {
        "LogisticRegression": ("Linear / probabilistic",
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
            "StandardScaler; L2, C=1.0, lbfgs", None),
        "GaussianNB": ("Linear / probabilistic",
            lambda: make_pipeline(StandardScaler(), GaussianNB()),
            "StandardScaler; var_smoothing=1e-9", None),
        "KNN": ("Instance-based",
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1)),
            "StandardScaler; k=50, distance-weighted, Euclidean; fit on 15k-point subsample", 15000),
        "DecisionTree": ("Tree-based",
            lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED),
            "max_depth=12, min_samples_leaf=50", None),
        "RandomForest": ("Ensemble (bagging)",
            lambda: RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20,
                                           max_features="sqrt", n_jobs=-1, random_state=SEED),
            "150 trees, max_depth=18, min_samples_leaf=20, sqrt features", None),
        "ExtraTrees": ("Ensemble (bagging)",
            lambda: ExtraTreesClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20,
                                         max_features="sqrt", n_jobs=-1, random_state=SEED),
            "150 trees, max_depth=18, min_samples_leaf=20, sqrt features", None),
        "HistGradientBoosting": ("Boosting",
            lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                                   random_state=SEED),
            "300 iters, lr=0.1, 31 leaves, no early stopping", None),
        "LightGBM": ("Boosting",
            lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1),
            "400 trees, lr=0.05, 63 leaves, bagging 0.8/0.8", None),
        "MLP": ("Neural network",
            lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                  batch_size=1024, learning_rate_init=1e-3, max_iter=40, early_stopping=True,
                                  validation_fraction=0.1, n_iter_no_change=5, random_state=SEED)),
            "StandardScaler; (64,32) ReLU, Adam lr=1e-3, batch 1024, alpha=1e-4, early stopping", None),
    }


def subsample_points(df: pd.DataFrame, max_points: int | None, seed=SEED) -> pd.DataFrame:
    if max_points is None:
        return df
    keys = df[["trip", "point"]].drop_duplicates()
    if len(keys) <= max_points:
        return df
    keys = keys.sample(max_points, random_state=seed)
    return df.merge(keys, on=["trip", "point"])


class ArcCache:
    """Per-trip arc table and canonical ids (for route-continuity metrics)."""
    def __init__(self, part):
        self.part, self._c = part, {}

    def get(self, tid):
        if tid not in self._c:
            arcs = load_trip(self.part, tid).arcs
            _, first, inv = np.unique(arcs, axis=0, return_index=True, return_inverse=True)
            self._c[tid] = (arcs, first[inv.ravel()])
        return self._c[tid]


def route_continuity(top: pd.DataFrame, cache: ArcCache) -> float:
    vals, w = [], []
    for tid, g in top.groupby("trip"):
        arcs, canon = cache.get(tid)
        vals.append(continuity_from_arcs(g.arc.to_numpy(), arcs, canon)); w.append(len(g) - 1)
    return float(np.average(vals, weights=w))


def evaluate(df, p, pts, cache):
    m = classification_metrics(df.label.to_numpy(), p)
    top = top1_predictions(df, p)
    m["top1_accuracy"] = top.label.sum() / len(pts)
    m["candidate_miss_rate"] = 1 - pts.cand_hit.mean()
    m["route_continuity"] = route_continuity(top, cache)
    return m, top


def main(only: list[str] | None = None):
    tr = pd.read_parquet(RESULTS_DIR / "features_train.parquet")
    te = pd.read_parquet(RESULTS_DIR / "features_student_test.parquet")
    pts_tr = pd.read_parquet(RESULTS_DIR / "points_train.parquet")
    pts_te = pd.read_parquet(RESULTS_DIR / "points_student_test.parquet")
    hit = pts_tr.loc[pts_tr.cand_hit == 1, ["trip", "point"]]
    trf = tr.merge(hit, on=["trip", "point"])
    print("train rows (filtered)", len(trf), "of", len(tr), "| test rows", len(te))
    cache_tr, cache_te = ArcCache("train"), ArcCache("student_test")
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(trf, trf.label, groups=trf.trip))

    summary, fold_rows = [], []
    log = open(OUT / "experiment_log.jsonl", "a", encoding="utf-8")
    zoo = model_zoo()
    for name, (family, make, cfg, maxpts) in zoo.items():
        if only and name not in only:
            continue
        print(f"\n=== {name}")
        cv = []
        for k, (itr, iva) in enumerate(folds):
            dtr = subsample_points(trf.iloc[itr], maxpts)
            dva = trf.iloc[iva]
            est = make(); t0 = time.time()
            est.fit(dtr[FEATURES].to_numpy(), dtr.label.to_numpy()); ft = time.time() - t0
            p = est.predict_proba(dva[FEATURES].to_numpy())[:, 1]
            pv = pts_tr[pts_tr.trip.isin(dva.trip.unique()) & (pts_tr.cand_hit == 1)]
            m, _ = evaluate(dva, p, pv, cache_tr)
            m.update(model=name, fold=k, fit_s=ft); m.pop("confusion")
            cv.append(m); fold_rows.append(m)
            print(f"  fold {k}: macroF1={m['macro_f1']:.4f} top1={m['top1_accuracy']:.4f} ll={m['log_loss']:.4f} fit={ft:.1f}s")
        cvdf = pd.DataFrame(cv)

        # refit on all (filtered) training data, evaluate on student_test
        dtr = subsample_points(trf, maxpts)
        est = make(); t0 = time.time()
        est.fit(dtr[FEATURES].to_numpy(), dtr.label.to_numpy()); fit_s = time.time() - t0
        t0 = time.time(); p = est.predict_proba(te[FEATURES].to_numpy())[:, 1]; inf_s = time.time() - t0
        mt, top = evaluate(te, p, pts_te, cache_te)
        trp = est.predict_proba(dtr[FEATURES].to_numpy())[:, 1]
        mtrain = classification_metrics(dtr.label.to_numpy(), trp)
        np.save(OUT / f"test_scores_{name}.npy", p.astype(np.float32))

        row = {"model": name, "family": family, "config": cfg,
               "train_rows": len(dtr),
               "cv_macro_f1_mean": cvdf.macro_f1.mean(), "cv_macro_f1_std": cvdf.macro_f1.std(),
               "cv_top1_mean": cvdf.top1_accuracy.mean(), "cv_top1_std": cvdf.top1_accuracy.std(),
               "cv_log_loss_mean": cvdf.log_loss.mean(), "cv_route_continuity": cvdf.route_continuity.mean(),
               "train_macro_f1": mtrain["macro_f1"], "train_log_loss": mtrain["log_loss"],
               "test_accuracy": mt["accuracy"], "test_precision": mt["precision"], "test_recall": mt["recall"],
               "test_f1": mt["f1"], "test_macro_f1": mt["macro_f1"], "test_log_loss": mt["log_loss"],
               "test_top1_accuracy": mt["top1_accuracy"], "test_candidate_miss_rate": mt["candidate_miss_rate"],
               "test_route_continuity": mt["route_continuity"],
               "fit_time_s": fit_s, "inference_time_s": inf_s,
               "inference_ms_per_1k_rows": inf_s / len(te) * 1e6,
               "test_confusion": mt["confusion"]}
        summary.append(row)
        log.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
                              **{k: (v if not isinstance(v, np.generic) else v.item()) for k, v in row.items()}}) + "\n")
        log.flush()
        print(f"  TEST macroF1={mt['macro_f1']:.4f} top1={mt['top1_accuracy']:.4f} ll={mt['log_loss']:.4f} "
              f"cont={mt['route_continuity']:.4f} fit={fit_s:.1f}s inf={inf_s:.2f}s")

        if name == "MLP":
            mlp = est[-1]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(mlp.loss_curve_, label="training log loss")
            ax2 = ax.twinx(); ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation accuracy")
            ax.set(xlabel="epoch", ylabel="training loss", title="MLP training history (final refit)")
            ax2.set_ylabel("validation accuracy (10% hold-out)")
            fig.legend(loc="center right"); fig.tight_layout(); fig.savefig(OUT / "mlp_training_history.png", dpi=130)
            plt.close(fig)
            pd.DataFrame({"epoch": range(1, len(mlp.loss_curve_) + 1), "train_loss": mlp.loss_curve_,
                          "val_accuracy": mlp.validation_scores_}).to_csv(OUT / "mlp_training_history.csv", index=False)

    s = pd.DataFrame(summary)
    prev = OUT / "model_comparison.csv"
    if only and prev.exists():
        old = pd.read_csv(prev)
        s = pd.concat([old[~old.model.isin(s.model)], s], ignore_index=True)
        f_old = pd.read_csv(OUT / "cv_folds.csv")
        fold_rows = f_old[~f_old.model.isin(only)].to_dict("records") + fold_rows
    s.to_csv(prev, index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "cv_folds.csv", index=False)
    plots(s, pd.DataFrame(fold_rows))


def plots(s: pd.DataFrame, f: pd.DataFrame):
    order = s.sort_values("test_top1_accuracy").model.tolist()
    s = s.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [f[f.model == m].macro_f1.values for m in order]
    ax.boxplot(data, vert=False, tick_labels=order)
    ax.set(xlabel="macro F1 (5-fold GroupKFold, train)", title="Cross-validation macro F1 by model")
    fig.tight_layout(); fig.savefig(OUT / "cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(order)); w = .38
    ax.barh(x - w / 2, s.test_top1_accuracy, w, label="Top-1 accuracy")
    ax.barh(x + w / 2, s.test_macro_f1, w, label="macro F1")
    for i, (a, b) in enumerate(zip(s.test_top1_accuracy, s.test_macro_f1)):
        ax.text(a, i - w / 2, f" {a:.3f}", va="center", fontsize=8); ax.text(b, i + w / 2, f" {b:.3f}", va="center", fontsize=8)
    ax.set_yticks(x, order); ax.set_xlim(0, 1.08); ax.legend(loc="lower right")
    ax.set_title("Student-test performance (classifier only)")
    fig.tight_layout(); fig.savefig(OUT / "student_test_top1_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].barh(order, s.test_log_loss); ax[0].set(xscale="log", title="Student-test log loss (lower is better)")
    ax[1].scatter(s.inference_ms_per_1k_rows, s.test_top1_accuracy)
    for m, r in s.iterrows():
        ax[1].annotate(m, (r.inference_ms_per_1k_rows, r.test_top1_accuracy), fontsize=8)
    ax[1].set(xscale="log", xlabel="inference ms / 1k candidate rows", ylabel="Top-1 accuracy",
              title="Accuracy vs inference cost")
    fig.tight_layout(); fig.savefig(OUT / "log_loss_and_speed.png", dpi=130); plt.close(fig)

    n = len(order); cols = 3; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(12, 3.6 * rows))
    for a, m in zip(axs.ravel(), order):
        cm = np.array(s.loc[m, "test_confusion"] if not isinstance(s.loc[m, "test_confusion"], str)
                      else json.loads(s.loc[m, "test_confusion"]))
        a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, f"{v}\n({v / cm[i].sum():.2f})", ha="center", va="center", fontsize=8)
        a.set(title=m, xticks=[0, 1], yticks=[0, 1], xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
    for a in axs.ravel()[n:]:
        a.axis("off")
    fig.suptitle("Student-test confusion matrices (candidate level, threshold 0.5)")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
