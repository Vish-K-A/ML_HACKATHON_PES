"""
Bayesian Hyperparameter Optimisation using Optuna (Tree-structured Parzen Estimator - TPE)
with Group-Aware Cross-Validation on the 80 Training Trips, followed by Final Model
Export and HMM + Viterbi Route Reconstruction Evaluation.
"""

import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from feature_engineering.features import FEATURE_NAMES
from hmm_viterbi.viterbi_decoder import HMMConfig, ViterbiRouteDecoder
from model_evaluation.metrics import (
    compute_reconstructed_route_continuity,
    compute_route_continuity,
    compute_sequence_overlap_metrics,
    evaluate_predictions_on_trips,
)
from model_training.train_and_compare import flatten_trip_cache, subsample_by_group

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_optimisation(
    train_cache: List[Dict[str, Any]],
    test_cache: List[Dict[str, Any]],
    output_dir: str,
    models_dir: str,
    n_trials: int = 20,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    X_train_all, y_train_all, groups_all = flatten_trip_cache(train_cache)
    K = train_cache[0]["X"].shape[1]

    # Use GroupKFold across the 80 training trips; subsample within folds for fast TPE search
    gkf = GroupKFold(n_splits=3)
    folds = []
    for f_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train_all, y_train_all, groups_all)):
        X_tr, y_tr = subsample_by_group(
            X_train_all[tr_idx], y_train_all[tr_idx], groups_all[tr_idx],
            max_samples=220000, seed=100 + f_idx
        )
        # Subsample validation fold by complete observation rows (multiples of K)
        n_val_obs = len(val_idx) // K
        val_2d_X = X_train_all[val_idx].reshape(n_val_obs, K, -1)
        val_2d_y = y_train_all[val_idx].reshape(n_val_obs, K)
        if n_val_obs > 12000:
            rng = np.random.default_rng(200 + f_idx)
            sel = np.sort(rng.choice(n_val_obs, size=12000, replace=False))
            val_2d_X = val_2d_X[sel]
            val_2d_y = val_2d_y[sel]
        folds.append((
            X_tr,
            y_tr,
            val_2d_X.reshape(-1, val_2d_X.shape[-1]),
            val_2d_y.reshape(-1),
            val_2d_y,
        ))

    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 220, 550),
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.14, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 24, 112),
            "max_depth": trial.suggest_int("max_depth", 6, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 8.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 2.4),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        fold_f1s = []
        fold_top1s = []
        fold_lls = []
        for X_tr, y_tr, X_val, y_val, y_val_2d in folds:
            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(np.int32)
            fold_f1s.append(float(f1_score(y_val, preds, average="macro", zero_division=0)))
            fold_lls.append(float(log_loss(y_val, np.clip(probs, 1e-6, 1.0 - 1e-6))))
            probs_2d = probs.reshape(-1, K)
            top1_ok = y_val_2d[np.arange(len(probs_2d)), np.argmax(probs_2d, axis=1)]
            fold_top1s.append(float(np.mean(top1_ok)))

        mean_f1 = float(np.mean(fold_f1s))
        mean_top1 = float(np.mean(fold_top1s))
        mean_ll = float(np.mean(fold_lls))
        elapsed = time.perf_counter() - t0

        trial_records.append({
            "trial": trial.number + 1,
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in params.items() if k not in ("random_state", "n_jobs", "verbose")},
            "cv_macro_f1": round(mean_f1, 4),
            "cv_top1_accuracy": round(mean_top1, 4),
            "cv_log_loss": round(mean_ll, 5),
            "trial_time_s": round(elapsed, 2),
        })
        print(
            f"  [Bayesian Trial {trial.number + 1:02d}/{n_trials}] "
            f"CV Macro F1={mean_f1:.4f} | CV Top-1={mean_top1:.4f} | "
            f"leaves={params['num_leaves']}, depth={params['max_depth']}, lr={params['learning_rate']:.4f} ({elapsed:.1f}s)"
        )
        return mean_f1

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="PathSnap_LightGBM_TPE")
    # Enqueue baseline configuration as Trial 1 for direct comparison
    study.enqueue_trial({
        "n_estimators": 250,
        "learning_rate": 0.07,
        "num_leaves": 31,
        "max_depth": 10,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "scale_pos_weight": 1.0,
    })

    t0_opt = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    total_opt_time = time.perf_counter() - t0_opt

    df_trials = pd.DataFrame(trial_records)
    df_trials.to_csv(os.path.join(output_dir, "bayesian_optimisation_trials.csv"), index=False)

    best_params = dict(study.best_params)
    best_params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})

    # Refit Untuned Baseline on full 80 training trips for exact side-by-side comparison
    untuned_params = {
        "n_estimators": 250,
        "learning_rate": 0.07,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    t0_un = time.perf_counter()
    clf_untuned = LGBMClassifier(**untuned_params)
    clf_untuned.fit(X_train_all, y_train_all)
    untuned_train_time = time.perf_counter() - t0_un

    t0_un_inf = time.perf_counter()
    untuned_probs_test = [
        clf_untuned.predict_proba(td["X"].reshape(-1, td["X"].shape[-1]))[:, 1].reshape(td["X"].shape[:2])
        for td in test_cache
    ]
    untuned_inf_time = time.perf_counter() - t0_un_inf
    untuned_eval = evaluate_predictions_on_trips(test_cache, untuned_probs_test, has_probabilities=True)

    # Refit Final Tuned Model on full 80 training trips
    t0_tu = time.perf_counter()
    clf_tuned = LGBMClassifier(**best_params)
    clf_tuned.fit(X_train_all, y_train_all)
    tuned_train_time = time.perf_counter() - t0_tu

    t0_tu_inf = time.perf_counter()
    tuned_probs_test = [
        clf_tuned.predict_proba(td["X"].reshape(-1, td["X"].shape[-1]))[:, 1].reshape(td["X"].shape[:2])
        for td in test_cache
    ]
    tuned_inf_time = time.perf_counter() - t0_tu_inf
    tuned_eval = evaluate_predictions_on_trips(test_cache, tuned_probs_test, has_probabilities=True)

    # Feature importances of final tuned model
    importances = clf_tuned.feature_importances_.tolist()
    df_imp = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance_splits": importances,
        "importance_gain": clf_tuned.booster_.feature_importance(importance_type="gain").tolist(),
    }).sort_values("importance_gain", ascending=False)
    df_imp.to_csv(os.path.join(output_dir, "feature_importances.csv"), index=False)

    # Evaluate HMM + Viterbi Route Reconstruction using the Final Tuned Model
    hmm_cfg = HMMConfig()
    decoder = ViterbiRouteDecoder(config=hmm_cfg)

    per_trip_rows = []
    total_obs = 0
    clf_top1_hits = 0
    hmm_top1_hits = 0
    clf_route_hits = 0
    hmm_route_hits = 0
    clf_conts = []
    hmm_conts = []
    hmm_strict_route_conts = []
    clf_seq_f1s = []
    hmm_seq_f1s = []
    clf_seq_jacc = []
    hmm_seq_jacc = []

    t0_hmm = time.perf_counter()
    for idx, tdata in enumerate(test_cache):
        probs_t = tuned_probs_test[idx]  # (T, K)
        cand_arcs = tdata["cand_arcs"]
        gt_arcs = tdata["gt_arcs"]
        arcs = tdata["arcs"]
        nodes_xy = tdata["nodes_xy"]
        route = tdata["route"]
        T = len(gt_arcs)
        route_set = set(map(int, route))

        # Classifier-only prediction
        clf_idx = np.argmax(probs_t, axis=1)
        clf_pred_arcs = cand_arcs[np.arange(T), clf_idx]

        # Classifier + HMM/Viterbi prediction
        dec_out = decoder.decode(
            arcs=arcs,
            nodes_xy=nodes_xy,
            cand_arcs=cand_arcs,
            probs=probs_t,
            perp_dist=tdata["perp_dist"],
            proj_frac=tdata["proj_frac"],
            cos_blend=tdata["cos_blend"],
            dt_prev=tdata["dt_prev"],
        )
        hmm_pred_arcs = dec_out["obs_pred_arcs"]
        rec_route = dec_out["reconstructed_route"]

        c_top1 = float(np.mean(clf_pred_arcs == gt_arcs))
        h_top1 = float(np.mean(hmm_pred_arcs == gt_arcs))
        c_in_r = float(np.mean([int(a) in route_set for a in clf_pred_arcs]))
        h_in_r = float(np.mean([int(a) in route_set for a in hmm_pred_arcs]))

        c_cont = compute_route_continuity(arcs, clf_pred_arcs)
        h_cont = compute_route_continuity(arcs, hmm_pred_arcs)
        h_rec_cont = compute_reconstructed_route_continuity(arcs, rec_route)

        c_seq = compute_sequence_overlap_metrics(clf_pred_arcs, route)
        h_seq = compute_sequence_overlap_metrics(rec_route, route)

        total_obs += T
        clf_top1_hits += int(np.sum(clf_pred_arcs == gt_arcs))
        hmm_top1_hits += int(np.sum(hmm_pred_arcs == gt_arcs))
        clf_route_hits += int(sum(1 for a in clf_pred_arcs if int(a) in route_set))
        hmm_route_hits += int(sum(1 for a in hmm_pred_arcs if int(a) in route_set))
        clf_conts.append(c_cont)
        hmm_conts.append(h_cont)
        hmm_strict_route_conts.append(h_rec_cont)
        clf_seq_f1s.append(c_seq["seq_f1"])
        hmm_seq_f1s.append(h_seq["seq_f1"])
        clf_seq_jacc.append(c_seq["seq_jaccard"])
        hmm_seq_jacc.append(h_seq["seq_jaccard"])

        per_trip_rows.append({
            "trip_id": tdata["trip_id"],
            "n_gps": T,
            "n_gt_route_arcs": len(route),
            "n_reconstructed_route_arcs": len(rec_route),
            "clf_top1_accuracy": round(c_top1, 4),
            "hmm_top1_accuracy": round(h_top1, 4),
            "top1_gain": round(h_top1 - c_top1, 4),
            "clf_route_match_acc": round(c_in_r, 4),
            "hmm_route_match_acc": round(h_in_r, 4),
            "clf_route_continuity": round(c_cont, 4),
            "hmm_route_continuity": round(h_cont, 4),
            "reconstructed_strict_continuity": round(h_rec_cont, 4),
            "clf_seq_f1": c_seq["seq_f1"],
            "hmm_seq_f1": h_seq["seq_f1"],
            "clf_seq_jaccard": c_seq["seq_jaccard"],
            "hmm_seq_jaccard": h_seq["seq_jaccard"],
        })
    hmm_decode_time = time.perf_counter() - t0_hmm

    df_per_trip = pd.DataFrame(per_trip_rows)
    df_per_trip.to_csv(os.path.join(output_dir, "per_trip_student_test_results.csv"), index=False)

    hmm_summary = {
        "classifier_only": {
            "observation_top1_accuracy": round(clf_top1_hits / max(1, total_obs), 4),
            "route_match_accuracy": round(clf_route_hits / max(1, total_obs), 4),
            "mean_route_continuity": round(float(np.mean(clf_conts)), 4),
            "mean_sequence_f1": round(float(np.mean(clf_seq_f1s)), 4),
            "mean_sequence_jaccard": round(float(np.mean(clf_seq_jacc)), 4),
            "inference_time_s": round(tuned_inf_time, 4),
        },
        "classifier_plus_hmm_viterbi": {
            "observation_top1_accuracy": round(hmm_top1_hits / max(1, total_obs), 4),
            "route_match_accuracy": round(hmm_route_hits / max(1, total_obs), 4),
            "mean_route_continuity": round(float(np.mean(hmm_conts)), 4),
            "reconstructed_strict_continuity": round(float(np.mean(hmm_strict_route_conts)), 4),
            "mean_sequence_f1": round(float(np.mean(hmm_seq_f1s)), 4),
            "mean_sequence_jaccard": round(float(np.mean(hmm_seq_jacc)), 4),
            "total_inference_and_decode_time_s": round(tuned_inf_time + hmm_decode_time, 4),
        },
    }

    # Save Final Single Model Artifact for Production / API Deployment
    final_artifact = {
        "model_name": "Tuned LightGBM + HMM-Viterbi (PathSnap Final Model)",
        "classifier": clf_tuned,
        "feature_names": FEATURE_NAMES,
        "k_candidates": K,
        "hmm_config": asdict(hmm_cfg),
        "best_hyperparameters": {k: v for k, v in best_params.items() if k not in ("random_state", "n_jobs", "verbose")},
        "student_test_summary": hmm_summary,
    }
    final_model_path = os.path.join(models_dir, "final_model.joblib")
    joblib.dump(final_artifact, final_model_path, compress=3)

    opt_summary = {
        "sampler": "Optuna TPESampler (Tree-structured Parzen Estimator, multivariate=True, seed=42)",
        "objective_metric": "5-Fold GroupKFold Cross-Validation Macro F1-score on 80 Training Trips",
        "n_trials": n_trials,
        "total_optimisation_time_s": round(total_opt_time, 2),
        "best_trial_number": int(study.best_trial.number + 1),
        "best_cv_macro_f1": round(float(study.best_value), 4),
        "baseline_cv_macro_f1": round(float(df_trials.iloc[0]["cv_macro_f1"]), 4),
        "best_hyperparameters": final_artifact["best_hyperparameters"],
        "untuned_vs_tuned": {
            "untuned": {
                "cv_macro_f1": round(float(df_trials.iloc[0]["cv_macro_f1"]), 4),
                "student_test_top1_accuracy": untuned_eval["top1_accuracy"],
                "student_test_route_match_accuracy": untuned_eval["route_match_accuracy"],
                "student_test_macro_f1": untuned_eval["macro_f1"],
                "student_test_log_loss": untuned_eval["log_loss"],
                "student_test_route_continuity": untuned_eval["route_continuity"],
                "training_time_s": round(untuned_train_time, 3),
                "inference_time_s": round(untuned_inf_time, 4),
            },
            "tuned": {
                "cv_macro_f1": round(float(study.best_value), 4),
                "student_test_top1_accuracy": tuned_eval["top1_accuracy"],
                "student_test_route_match_accuracy": tuned_eval["route_match_accuracy"],
                "student_test_macro_f1": tuned_eval["macro_f1"],
                "student_test_log_loss": tuned_eval["log_loss"],
                "student_test_route_continuity": tuned_eval["route_continuity"],
                "training_time_s": round(tuned_train_time, 3),
                "inference_time_s": round(tuned_inf_time, 4),
            },
        },
        "hmm_viterbi_comparison": hmm_summary,
        "final_model_artifact_path": final_model_path,
    }

    with open(os.path.join(output_dir, "bayesian_optimisation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(opt_summary, f, indent=2)

    return opt_summary
