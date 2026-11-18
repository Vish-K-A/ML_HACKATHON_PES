"""Build candidate features and train Extra Trees and Logistic Regression.

Run from the repository root with: python -m src.model.train
This script uses only ../data/train and never loads ../data/test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.pathsnap import load_trip
from src.features.candidates import FEATURES, trip_candidates

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    config = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    data_root = (ROOT / config["data_root"]).resolve()
    trip_dirs = sorted((data_root / config["train_split"]).iterdir())
    trip_dirs = [p for p in trip_dirs if p.is_dir()]
    if len(trip_dirs) < 5:
        raise RuntimeError(f"Expected multiple training trips under {data_root / config['train_split']}")

    frames = []
    summaries = []
    started = time.perf_counter()
    for i, folder in enumerate(trip_dirs, start=1):
        trip = load_trip(folder)
        frame, summary = trip_candidates(
            trip,
            max_candidates=config["max_candidates"],
            radius_m=config["candidate_radius_m"],
            midpoint_prefilter=config["midpoint_prefilter"],
            max_negatives=config["max_negatives_per_observation"],
        )
        frames.append(frame)
        summaries.append({"trip_id": trip.trip_id, **summary})
        print(f"Prepared {i}/{len(trip_dirs)} trips: {trip.trip_id}", flush=True)

    data = pd.concat(frames, ignore_index=True)
    group_split = GroupShuffleSplit(
        n_splits=1, test_size=config["validation_fraction"], random_state=config["random_seed"]
    )
    train_idx, valid_idx = next(group_split.split(data, data["label"], groups=data["trip_id"]))
    train_data, valid_data = data.iloc[train_idx], data.iloc[valid_idx]
    models = {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=config["n_estimators"],
            min_samples_leaf=config["min_samples_leaf"],
            class_weight="balanced",
            random_state=config["random_seed"],
            n_jobs=config["n_jobs"],
        ),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=config["logistic_max_iter"],
                class_weight="balanced",
                solver="lbfgs",
                random_state=config["random_seed"],
            ),
        ),
    }
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "experiments").mkdir(exist_ok=True)
    model_metrics = {}
    for name, model in models.items():
        fit_started = time.perf_counter()
        model.fit(train_data[FEATURES], train_data["label"])
        valid = valid_data.copy()
        valid["score"] = model.predict_proba(valid[FEATURES])[:, 1]
        chosen = valid.loc[valid.groupby(["trip_id", "observation"])["score"].idxmax()]
        model_metrics[name] = {
            "observation_top1_accuracy": float(accuracy_score(chosen["reference_arc_id"], chosen["arc_id"])),
            "observation_macro_f1": float(f1_score(chosen["reference_arc_id"], chosen["arc_id"], average="macro", zero_division=0)),
            "fit_seconds": round(time.perf_counter() - fit_started, 2),
        }
        joblib.dump(
            {"model": model, "features": FEATURES, "config": config},
            ROOT / "artifacts" / f"{name}.joblib",
        )

    candidate_recall = float(np.mean([
        summary["candidate_recall"] for summary in summaries
        if summary["trip_id"] in set(valid_data["trip_id"])
    ]))
    metrics = {
        "candidate_rows_train": int(len(train_data)),
        "candidate_rows_validation": int(len(valid_data)),
        "validation_trip_ids": sorted(valid_data["trip_id"].unique().tolist()),
        "models": model_metrics,
        "validation_candidate_recall": candidate_recall,
        "median_alignment_error_m": float(np.median([s["alignment_median_m"] for s in summaries])),
        "p95_alignment_error_m": float(np.percentile([s["alignment_p95_m"] for s in summaries], 95)),
        "feature_names": FEATURES,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    pd.DataFrame(summaries).to_csv(ROOT / "experiments" / "trip_diagnostics.csv", index=False)
    (ROOT / "experiments" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
