"""Select, tune, and HMM-decode a candidate road scorer.

Run this AFTER ``python -m src.model.train`` has created the baseline model
artifacts and experiments/metrics.json. It writes only HMM-specific outputs.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.pathsnap import load_trip, project_trip
from src.features.candidates import FEATURES, trip_candidates

ROOT = Path(__file__).resolve().parents[2]
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_model(name: str, params: dict, config: dict):
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", config["n_estimators"])),
            min_samples_leaf=int(params.get("min_samples_leaf", config["min_samples_leaf"])),
            class_weight="balanced",
            random_state=config["random_seed"],
            n_jobs=config["n_jobs"],
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=config["logistic_max_iter"],
            class_weight="balanced",
            solver="lbfgs",
            random_state=config["random_seed"],
        ),
    )


def top1_rows(frame: pd.DataFrame, model) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = model.predict_proba(scored[FEATURES])[:, 1]
    return scored.loc[scored.groupby(["trip_id", "observation"])["score"].idxmax()].copy()


def top1_metrics(chosen: pd.DataFrame) -> dict[str, float]:
    return {
        "observation_top1_accuracy": float(accuracy_score(chosen["reference_arc_id"], chosen["arc_id"])),
        "observation_macro_f1": float(f1_score(
            chosen["reference_arc_id"], chosen["arc_id"], average="macro", zero_division=0
        )),
    }


def viterbi_decode(
    candidates: pd.DataFrame,
    trip,
    nodes_xy: np.ndarray,
    track_xy: np.ndarray,
    distance_scale_m: float,
    disconnected_penalty: float,
) -> pd.DataFrame:
    """Decode one trip's per-point candidate scores with a Viterbi recurrence."""
    emissions = []
    backpointers = []
    states = []
    previous_midpoints = None
    previous_to_nodes = None
    previous_observation = None

    for observation, group in candidates.groupby("observation", sort=True):
        group = group.reset_index(drop=True)
        arc_ids = group["arc_id"].to_numpy(dtype=np.int64)
        arc_endpoints = trip.arcs[arc_ids]
        starts = nodes_xy[arc_endpoints[:, 0]]
        ends = nodes_xy[arc_endpoints[:, 1]]
        midpoints = (starts + ends) / 2.0
        from_nodes = arc_endpoints[:, 0]
        to_nodes = arc_endpoints[:, 1]
        emission = np.log(np.clip(group["score"].to_numpy(dtype=float), 1e-12, 1.0))

        if previous_midpoints is None:
            delta = emission
            back = np.full(len(group), -1, dtype=np.int64)
        else:
            gps_step = float(np.linalg.norm(track_xy[int(observation)] - track_xy[previous_observation]))
            move = np.linalg.norm(previous_midpoints[:, None, :] - midpoints[None, :, :], axis=2)
            transition = -np.abs(move - gps_step) / max(distance_scale_m, 1e-6)
            connected = previous_to_nodes[:, None] == from_nodes[None, :]
            transition -= (~connected) * disconnected_penalty
            scores = delta[:, None] + transition
            back = np.argmax(scores, axis=0).astype(np.int64)
            delta = scores[back, np.arange(len(group))] + emission

        states.append(group)
        emissions.append(delta)
        backpointers.append(back)
        previous_midpoints = midpoints
        previous_to_nodes = to_nodes
        previous_observation = int(observation)

    if not states:
        return pd.DataFrame()
    current = int(np.argmax(emissions[-1]))
    chosen_indices = [current]
    for t in range(len(states) - 1, 0, -1):
        current = int(backpointers[t][current])
        chosen_indices.append(current)
    chosen_indices.reverse()
    return pd.concat([states[t].iloc[[idx]] for t, idx in enumerate(chosen_indices)], ignore_index=True)


def tune_model(name: str, train_data: pd.DataFrame, config: dict, hmm_config: dict):
    groups = train_data["trip_id"].to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(int(hmm_config["bayes_cv_folds"]), len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("Not enough training trips for grouped hyperparameter tuning")
    splitter = GroupKFold(n_splits=n_splits)

    def objective(trial):
        if name == "extra_trees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 60, 140, step=40),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }
        else:
            params = {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
        scores = []
        for fit_idx, val_idx in splitter.split(train_data, groups=groups):
            fit_rows = train_data.iloc[fit_idx]
            val_rows = train_data.iloc[val_idx]
            if fit_rows["label"].nunique() < 2:
                continue
            model = make_model(name, params, config)
            model.fit(fit_rows[FEATURES], fit_rows["label"])
            scores.append(top1_metrics(top1_rows(val_rows, model))["observation_top1_accuracy"])
        if not scores:
            raise RuntimeError("A tuning fold did not contain both candidate classes")
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["random_seed"]),
    )
    study.optimize(
        objective,
        n_trials=int(hmm_config["bayes_trials"]),
        timeout=int(hmm_config["bayes_timeout_seconds"]),
        show_progress_bar=True,
    )
    return study


def main() -> None:
    config = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    hmm_config = json.loads((ROOT / "configs" / "hmm_pipeline.json").read_text(encoding="utf-8"))
    baseline_path = ROOT / "experiments" / "metrics.json"
    if not baseline_path.exists():
        raise RuntimeError("Run the existing baseline pipeline first; experiments/metrics.json is missing.")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_results = baseline.get("models", {})
    available_models = [
        name for name in ("extra_trees", "logistic_regression")
        if (ROOT / "artifacts" / f"{name}.joblib").exists()
    ]
    if base_results:
        selected_name = max(
            base_results,
            key=lambda key: (
                base_results[key].get("observation_top1_accuracy", -1),
                base_results[key].get("observation_macro_f1", -1),
            ),
        )
    elif len(available_models) == 1:
        # Older baseline runs recorded only one model and top-level metrics.
        # Use that sole artifact instead of forcing a full baseline rerun.
        selected_name = available_models[0]
        print(
            f"Baseline metrics have no model-comparison section; using the only available artifact: {selected_name}",
            flush=True,
        )
    elif "observation_top1_accuracy" in baseline and "extra_trees" in available_models:
        selected_name = "extra_trees"
        print("Using the legacy Extra Trees baseline artifact.", flush=True)
    else:
        raise RuntimeError(
            "Baseline metrics lack model comparisons and no unambiguous baseline artifact is available. "
            "Ensure the baseline artifacts and metrics.json are from the same run."
        )
    artifact_path = ROOT / "artifacts" / f"{selected_name}.joblib"
    if not artifact_path.exists():
        raise RuntimeError(f"Selected baseline artifact is missing: {artifact_path.name}")

    data_root = (ROOT / config["data_root"]).resolve()
    trip_dirs = sorted(p for p in (data_root / config["train_split"]).iterdir() if p.is_dir())
    frames = []
    trip_summaries = {}
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
        trip_summaries[trip.trip_id] = summary
        print(f"Prepared HMM/tuning features {i}/{len(trip_dirs)}: {trip.trip_id}", flush=True)
    data = pd.concat(frames, ignore_index=True)
    split = GroupShuffleSplit(
        n_splits=1, test_size=config["validation_fraction"], random_state=config["random_seed"]
    )
    train_idx, valid_idx = next(split.split(data, data["label"], groups=data["trip_id"]))
    train_data = data.iloc[train_idx].copy()
    valid_data = data.iloc[valid_idx].copy()

    study = tune_model(selected_name, train_data, config, hmm_config)
    tuned = make_model(selected_name, study.best_params, config)
    tuned.fit(train_data[FEATURES], train_data["label"])
    chosen = top1_rows(valid_data, tuned)

    # Reconstruct only the held-out training trips; the test partition is never opened.
    hmm_rows = []
    continuity_values = []
    for trip_id, group in valid_data.groupby("trip_id", sort=True):
        trip = load_trip(data_root / config["train_split"] / trip_id)
        nodes_xy, track_xy = project_trip(trip)
        scored = group.copy()
        scored["score"] = tuned.predict_proba(scored[FEATURES])[:, 1]
        decoded = viterbi_decode(
            scored,
            trip,
            nodes_xy,
            track_xy,
            hmm_config["hmm_distance_scale_m"],
            hmm_config["hmm_disconnected_penalty"],
        )
        hmm_rows.append(decoded)
        decoded = decoded.sort_values("observation")
        if len(decoded) > 1:
            continuity_values.extend(
                (decoded["arc_id"].to_numpy()[1:] >= 0) &
                (trip.arcs[decoded["arc_id"].to_numpy()[:-1], 1] ==
                 trip.arcs[decoded["arc_id"].to_numpy()[1:], 0])
            )
    decoded_all = pd.concat(hmm_rows, ignore_index=True)
    valid_metrics = top1_metrics(chosen)
    hmm_metrics = top1_metrics(decoded_all)
    metrics = {
        "selected_model": selected_name,
        "selection_reason": "highest grouped-holdout observation Top-1 accuracy, with Macro F1 as tie-breaker",
        "baseline_model_comparison": base_results or {selected_name: baseline},
        "bayesian_trials_completed": len(study.trials),
        "bayesian_best_cv_top1_accuracy": float(study.best_value),
        "bayesian_best_params": study.best_params,
        "tuned_classifier_only": valid_metrics,
        "tuned_classifier_plus_hmm": {
            **hmm_metrics,
            "arc_transition_continuity": float(np.mean(continuity_values)) if continuity_values else None,
        },
        "validation_trip_ids": sorted(valid_data["trip_id"].unique().tolist()),
        "validation_candidate_recall": float(np.mean([
            trip_summaries[tid]["candidate_recall"] for tid in valid_data["trip_id"].unique()
        ])),
        "label_alignment_median_m": float(np.median([s["alignment_median_m"] for s in trip_summaries.values()])),
        "label_alignment_p95_m": float(np.percentile([s["alignment_p95_m"] for s in trip_summaries.values()], 95)),
        "hmm_transition_config": {
            "distance_scale_m": hmm_config["hmm_distance_scale_m"],
            "disconnected_penalty": hmm_config["hmm_disconnected_penalty"],
        },
    }
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "experiments").mkdir(exist_ok=True)
    joblib.dump(
        {"model": tuned, "model_name": selected_name, "features": FEATURES,
         "params": study.best_params, "config": config},
        ROOT / "artifacts" / "hmm_selected_tuned.joblib",
    )
    decoded_all[["trip_id", "observation", "reference_arc_id", "arc_id", "score"]].to_csv(
        ROOT / "experiments" / "hmm_predictions.csv", index=False
    )
    (ROOT / "experiments" / "hmm_pipeline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
