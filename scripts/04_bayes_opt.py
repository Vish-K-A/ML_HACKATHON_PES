"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model.

Usage: python scripts/04_bayes_opt.py [MODEL] [N_TRIALS]

Objective : maximise mean observation-level Top-1 accuracy over a 3-fold
            GroupKFold (by trip) on the 80 training trips. Top-1 is the primary
            selection metric because it directly measures map-matching quality;
            macro F1 and log loss are logged per trial as secondary attributes.
Cost      : inside the search the training folds are subsampled to 40 % of
            observations (whole observations, all their candidates).
Afterwards: untuned and tuned configurations are both re-evaluated with the
            identical 5-fold GroupKFold used in 03_train_models.py, refit on all
            training data and scored on student_test.
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
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.config import RESULTS_DIR, SEED
from pathsnap.features import FEATURES
from pathsnap.metrics import classification_metrics, top1_predictions

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
tm = import_module("03_train_models")

OUT = RESULTS_DIR / "optuna"
OUT.mkdir(parents=True, exist_ok=True)

SPACES = {
    "LightGBM": {
        "untuned": dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8),
        "suggest": lambda tr: dict(
            n_estimators=tr.suggest_int("n_estimators", 100, 1000, step=50),
            learning_rate=tr.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=tr.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=tr.suggest_int("min_child_samples", 10, 500, log=True),
            subsample=tr.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=tr.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            reg_alpha=tr.suggest_float("reg_alpha", 1e-3, 10.0, log=True)),
        "make": lambda p: LGBMClassifier(**p, random_state=SEED, verbose=-1, n_jobs=-1),
    },
    "HistGradientBoosting": {
        "untuned": dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31),
        "suggest": lambda tr: dict(
            max_iter=tr.suggest_int("max_iter", 100, 800, step=50),
            learning_rate=tr.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_leaf_nodes=tr.suggest_int("max_leaf_nodes", 15, 255, log=True),
            min_samples_leaf=tr.suggest_int("min_samples_leaf", 10, 500, log=True),
            l2_regularization=tr.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            max_features=tr.suggest_float("max_features", 0.4, 1.0)),
        "make": lambda p: HistGradientBoostingClassifier(**p, random_state=SEED),
    },
}


def cv_eval(make, params, trf, pts, folds, frac=None, seed=SEED):
    res = []
    for itr, iva in folds:
        dtr = trf.iloc[itr]
        if frac:
            keys = dtr[["trip", "point"]].drop_duplicates().sample(frac=frac, random_state=seed)
            dtr = dtr.merge(keys, on=["trip", "point"])
        dva = trf.iloc[iva]
        est = make(params); t0 = time.time()
        est.fit(dtr[FEATURES].to_numpy(), dtr.label.to_numpy()); ft = time.time() - t0
        p = est.predict_proba(dva[FEATURES].to_numpy())[:, 1]
        m = classification_metrics(dva.label.to_numpy(), p); m.pop("confusion")
        top = top1_predictions(dva, p)
        m["top1_accuracy"] = top.label.mean(); m["fit_s"] = ft
        res.append(m)
    return pd.DataFrame(res)


def main(model="LightGBM", n_trials=40):
    sp = SPACES[model]
    tr = pd.read_parquet(RESULTS_DIR / "features_train.parquet")
    te = pd.read_parquet(RESULTS_DIR / "features_student_test.parquet")
    pts_tr = pd.read_parquet(RESULTS_DIR / "points_train.parquet")
    pts_te = pd.read_parquet(RESULTS_DIR / "points_student_test.parquet")
    trf = tr.merge(pts_tr.loc[pts_tr.cand_hit == 1, ["trip", "point"]], on=["trip", "point"])
    folds3 = list(GroupKFold(n_splits=3).split(trf, trf.label, groups=trf.trip))

    def objective(trial):
        params = sp["suggest"](trial)
        r = cv_eval(sp["make"], params, trf, pts_tr, folds3, frac=0.4)
        for k in ["macro_f1", "log_loss", "fit_s"]:
            trial.set_user_attr(k, float(r[k].mean()))
        trial.set_user_attr("top1_fold_std", float(r.top1_accuracy.std()))
        return float(r.top1_accuracy.mean())

    storage = f"sqlite:///{(OUT / f'study_{model}.db').as_posix()}"
    study = optuna.create_study(direction="maximize", study_name=f"pathsnap_{model}", storage=storage,
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                load_if_exists=True)
    # first trial = untuned configuration so the search starts from the baseline
    if len(study.trials) == 0:
        study.enqueue_trial({k: v for k, v in sp["untuned"].items() if k != "subsample_freq"})
    t0 = time.time()
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    search_s = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(OUT / f"trials_{model}.csv", index=False)
    best = {**study.best_params}
    if model == "LightGBM":
        best["subsample_freq"] = 1
    print("best", study.best_value, best)

    # ------------- fair comparison: untuned vs tuned, identical protocol
    folds5 = list(GroupKFold(n_splits=5).split(trf, trf.label, groups=trf.trip))
    cache = tm.ArcCache("student_test")
    rows = {}
    fitted = {}
    for tag, params in [("untuned", sp["untuned"]), ("tuned", best)]:
        cv = cv_eval(sp["make"], params, trf, pts_tr, folds5)
        est = sp["make"](params); t0 = time.time()
        est.fit(trf[FEATURES].to_numpy(), trf.label.to_numpy()); fit_s = time.time() - t0
        t0 = time.time(); p = est.predict_proba(te[FEATURES].to_numpy())[:, 1]; inf_s = time.time() - t0
        mt, _ = tm.evaluate(te, p, pts_te, cache)
        rows[tag] = {"params": json.dumps(params), "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
                     "cv_top1": cv.top1_accuracy.mean(), "cv_log_loss": cv.log_loss.mean(),
                     "test_top1_accuracy": mt["top1_accuracy"], "test_macro_f1": mt["macro_f1"],
                     "test_log_loss": mt["log_loss"], "test_route_continuity": mt["route_continuity"],
                     "fit_time_s": fit_s, "inference_time_s": inf_s}
        fitted[tag] = (est, p)
        print(tag, {k: v for k, v in rows[tag].items() if k != "params"})
    comp = pd.DataFrame(rows).T
    comp["search_time_s"] = [np.nan, search_s]
    comp["n_trials"] = [np.nan, len(study.trials)]
    comp.to_csv(OUT / f"tuned_vs_untuned_{model}.csv")
    import joblib
    joblib.dump({"model": fitted["tuned"][0], "params": best, "model_name": model},
                RESULTS_DIR.parent / "models" / f"{model}_tuned.joblib")
    joblib.dump({"model": fitted["untuned"][0], "params": sp["untuned"], "model_name": model},
                RESULTS_DIR.parent / "models" / f"{model}_untuned.joblib")
    np.save(OUT / f"test_scores_{model}_tuned.npy", fitted["tuned"][1].astype(np.float32))

    # plots
    tdf = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(tdf.number, tdf.value, "o", alpha=.6, label="trial")
    ax[0].plot(tdf.number, tdf.value.cummax(), "r-", label="best so far")
    ax[0].set(xlabel="trial", ylabel="3-fold CV Top-1 accuracy", title=f"Optuna TPE search ({model})"); ax[0].legend()
    cmp = comp[["cv_macro_f1", "test_top1_accuracy", "test_macro_f1"]].astype(float)
    cmp.T.plot.bar(ax=ax[1], rot=0)
    lo = cmp.values.min()
    ax[1].set(ylim=(lo - 0.02, min(1, cmp.values.max() + 0.01)), title="Untuned vs tuned")
    fig.tight_layout(); fig.savefig(OUT / f"optuna_{model}.png", dpi=130); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1])
        ax.set(title="Hyperparameter importance (fANOVA)"); fig.tight_layout()
        fig.savefig(OUT / f"param_importance_{model}.png", dpi=130); plt.close(fig)
        pd.Series(imp).to_csv(OUT / f"param_importance_{model}.csv")
    except Exception as e:  # noqa: BLE001
        print("importance failed", e)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "LightGBM", int(sys.argv[2]) if len(sys.argv) > 2 else 40)
