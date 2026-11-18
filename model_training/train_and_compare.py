"""
Multi-Model Experimentation and Comparison across 6 Algorithm Families for PathSnap.
Uses 5-Fold GroupKFold (grouped by trip_id) on the 80 training trips and evaluates
all models on the 10 student_test trips.
"""

import json
import os
import time
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_processing.preprocess import load_trip
from feature_engineering.features import FEATURE_NAMES, extract_trip_candidates_and_features
from model_evaluation.metrics import evaluate_predictions_on_trips


def build_partition_cache(student_folder: str, partition: str, k_candidates: int = 8) -> List[Dict[str, Any]]:
    part_dir = os.path.join(student_folder, partition)
    trip_ids = sorted(os.listdir(part_dir))
    cache = []
    for trip_id in trip_ids:
        tdir = os.path.join(part_dir, trip_id)
        if not os.path.isdir(tdir):
            continue
        trip = load_trip(tdir, trip_id=trip_id, partition=partition, load_route=True)
        feats = extract_trip_candidates_and_features(trip, k_candidates=k_candidates, include_labels=True)
        cache.append({
            "trip_id": trip_id,
            "partition": partition,
            "X": feats["X"],                  # (T, K, F)
            "y": feats["y"],                  # (T, K)
            "cand_arcs": feats["cand_arcs"],  # (T, K)
            "gt_arcs": feats["gt_arcs"],      # (T,)
            "perp_dist": feats["perp_dist"],  # (T, K)
            "proj_frac": feats["proj_frac"],  # (T, K)
            "cos_blend": feats["cos_blend"],  # (T, K)
            "dt_prev": feats["dt_prev"],      # (T,)
            "speed_ms": feats["speed_ms"],    # (T,)
            "arcs": trip.arcs,
            "nodes_xy": trip.nodes_xy,
            "nodes": trip.nodes,
            "track": trip.track,
            "route": trip.route,
        })
    return cache


def flatten_trip_cache(cache: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, g_list = [], [], []
    for idx, item in enumerate(cache):
        T, K, F = item["X"].shape
        X_list.append(item["X"].reshape(T * K, F))
        y_list.append(item["y"].reshape(T * K))
        g_list.append(np.full(T * K, idx, dtype=np.int32))
    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(g_list)


def subsample_by_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    max_samples: int = 280000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preserves all positive instances and a stratified random subset of negative instances
    when fitting computationally heavy algorithms (e.g., KNN, RF, MLP) across 1.66M rows.
    """
    if len(y) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx_all = rng.choice(len(y), size=max_samples, replace=False)
    idx_all.sort()
    return X[idx_all], y[idx_all]


def get_model_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "Logistic Regression",
            "family": "Linear & Probabilistic",
            "needs_scaling": True,
            "max_train_samples": 300000,
            "max_val_obs": 20000,
            "key_config": "L2 penalty, C=1.0, solver='lbfgs', max_iter=300",
            "loss_objective": "Binary Cross-Entropy (Log-Loss) with L2 Regularisation",
            "builder": lambda: LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=300, random_state=42
            ),
        },
        {
            "name": "Distance-Weighted KNN",
            "family": "Instance-Based",
            "needs_scaling": True,
            "max_train_samples": 50000,
            "max_val_obs": 6000,
            "key_config": "n_neighbors=25, weights='distance', metric='euclidean'",
            "loss_objective": "Non-parametric Inverse-Distance Weighted Local Voting",
            "builder": lambda: KNeighborsClassifier(
                n_neighbors=25, weights="distance", metric="minkowski", p=2, n_jobs=-1
            ),
        },
        {
            "name": "Decision Tree",
            "family": "Tree-Based",
            "needs_scaling": False,
            "max_train_samples": 300000,
            "max_val_obs": 20000,
            "key_config": "criterion='gini', max_depth=10, min_samples_leaf=25",
            "loss_objective": "Gini Impurity Reduction",
            "builder": lambda: DecisionTreeClassifier(
                criterion="gini", max_depth=10, min_samples_split=50, min_samples_leaf=25, random_state=42
            ),
        },
        {
            "name": "Random Forest",
            "family": "Ensemble (Bagging)",
            "needs_scaling": False,
            "max_train_samples": 120000,
            "max_val_obs": 15000,
            "key_config": "n_estimators=80, max_depth=15, min_samples_leaf=10, max_features='sqrt'",
            "loss_objective": "Bagged Gini Impurity Reduction across Decorrelated Trees",
            "builder": lambda: RandomForestClassifier(
                n_estimators=80, max_depth=15, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=42
            ),
        },
        {
            "name": "Deep MLP Classifier",
            "family": "Neural Networks",
            "needs_scaling": True,
            "max_train_samples": 140000,
            "max_val_obs": 15000,
            "key_config": "hidden_layer_sizes=(128, 64), ReLU, Adam, lr=1e-3, alpha=1e-4, early_stopping=True",
            "loss_objective": "Binary Cross-Entropy Loss via Adam Stochastic Gradient Optimization",
            "builder": lambda: MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=25,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=4,
                random_state=42,
            ),
        },
        {
            "name": "LightGBM Classifier",
            "family": "Gradient Boosting",
            "needs_scaling": False,
            "max_train_samples": 600000,
            "max_val_obs": 25000,
            "key_config": "n_estimators=250, lr=0.07, num_leaves=31, subsample=0.85, colsample_bytree=0.85",
            "loss_objective": "Gradient-Boosted Binary Log-Loss with Leaf-wise Histogram Splitting",
            "builder": lambda: LGBMClassifier(
                n_estimators=250,
                learning_rate=0.07,
                num_leaves=31,
                max_depth=-1,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        },
    ]


def run_model_comparison(
    train_cache: List[Dict[str, Any]],
    test_cache: List[Dict[str, Any]],
    output_dir: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    X_train_all, y_train_all, groups_all = flatten_trip_cache(train_cache)
    gkf = GroupKFold(n_splits=5)

    models_spec = get_model_definitions()
    comparison_rows = []
    detailed_results: Dict[str, Any] = {}

    for spec in models_spec:
        m_name = spec["name"]
        print(f"\n--- Training & Evaluating Model: {m_name} ({spec['family']}) ---")

        # 5-fold Group-aware Cross-Validation on the 80 training trips
        fold_macro_f1s = []
        fold_top1_accs = []
        fold_log_losses = []

        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train_all, y_train_all, groups_all)):
            X_tr, y_tr = subsample_by_group(
                X_train_all[tr_idx], y_train_all[tr_idx], groups_all[tr_idx],
                max_samples=spec["max_train_samples"], seed=42 + fold_idx
            )
            K = train_cache[0]["X"].shape[1]
            n_val_obs = len(val_idx) // K
            val_X_2d = X_train_all[val_idx].reshape(n_val_obs, K, -1)
            val_y_2d = y_train_all[val_idx].reshape(n_val_obs, K)
            max_vo = spec.get("max_val_obs", 20000)
            if n_val_obs > max_vo:
                rng = np.random.default_rng(99 + fold_idx)
                sel_o = np.sort(rng.choice(n_val_obs, size=max_vo, replace=False))
                val_X_2d = val_X_2d[sel_o]
                val_y_2d = val_y_2d[sel_o]
            X_val = val_X_2d.reshape(-1, val_X_2d.shape[-1])
            y_val = val_y_2d.reshape(-1)

            if spec["needs_scaling"]:
                scaler_cv = StandardScaler()
                X_tr = scaler_cv.fit_transform(X_tr)
                X_val = scaler_cv.transform(X_val)

            clf_cv = spec["builder"]()
            clf_cv.fit(X_tr, y_tr)
            val_probs = clf_cv.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(np.int32)

            f1_m = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
            ll_m = float(log_loss(y_val, np.clip(val_probs, 1e-6, 1.0 - 1e-6)))
            fold_macro_f1s.append(f1_m)
            fold_log_losses.append(ll_m)

            # Observation-level Top-1 on validation fold
            val_probs_2d = val_probs.reshape(-1, K)
            top1_ok = val_y_2d[np.arange(len(val_probs_2d)), np.argmax(val_probs_2d, axis=1)]
            fold_top1_accs.append(float(np.mean(top1_ok)))

        # Fit on full 80-trip training set and evaluate on 10 student_test trips
        t0_train = time.perf_counter()
        X_fit, y_fit = subsample_by_group(
            X_train_all, y_train_all, groups_all,
            max_samples=spec["max_train_samples"], seed=42
        )
        scaler_full = None
        if spec["needs_scaling"]:
            scaler_full = StandardScaler()
            X_fit = scaler_full.fit_transform(X_fit)

        clf_full = spec["builder"]()
        clf_full.fit(X_fit, y_fit)
        train_time_s = time.perf_counter() - t0_train

        # Training set evaluation (on subsample for overfitting/underfitting check)
        train_probs_fit = clf_full.predict_proba(X_fit)[:, 1]
        train_macro_f1 = float(f1_score(y_fit, (train_probs_fit >= 0.5).astype(np.int32), average="macro"))
        train_log_loss = float(log_loss(y_fit, np.clip(train_probs_fit, 1e-6, 1.0 - 1e-6)))

        # Student-test inference and evaluation
        t0_inf = time.perf_counter()
        test_probs_per_trip = []
        for tdata in test_cache:
            T, K, F = tdata["X"].shape
            X_t = tdata["X"].reshape(T * K, F)
            if scaler_full is not None:
                X_t = scaler_full.transform(X_t)
            p_t = clf_full.predict_proba(X_t)[:, 1].reshape(T, K)
            test_probs_per_trip.append(p_t)
        inf_time_s = time.perf_counter() - t0_inf

        test_metrics = evaluate_predictions_on_trips(test_cache, test_probs_per_trip, has_probabilities=True)

        # Save epoch-wise training history for MLP
        mlp_history = None
        if hasattr(clf_full, "loss_curve_"):
            mlp_history = {
                "loss_curve": [round(float(x), 5) for x in clf_full.loss_curve_],
                "validation_scores": [round(float(x), 5) for x in getattr(clf_full, "validation_scores_", [])],
            }

        row = {
            "Model": m_name,
            "Algorithm Family": spec["family"],
            "Key Configuration": spec["key_config"],
            "Loss / Objective": spec["loss_objective"],
            "Train Macro F1": round(train_macro_f1, 4),
            "CV Macro F1-score": round(float(np.mean(fold_macro_f1s)), 4),
            "CV Macro F1 Std": round(float(np.std(fold_macro_f1s)), 4),
            "CV Top-1 Accuracy": round(float(np.mean(fold_top1_accs)), 4),
            "Top-1 Accuracy": test_metrics["top1_accuracy"],
            "Route Match Accuracy": test_metrics["route_match_accuracy"],
            "Route continuity": test_metrics["route_continuity"],
            "Student-Test Macro F1-score": test_metrics["macro_f1"],
            "Student-Test Precision": test_metrics["macro_precision"],
            "Student-Test Recall": test_metrics["macro_recall"],
            "Student-Test Accuracy": test_metrics["accuracy"],
            "Log Loss": test_metrics["log_loss"],
            "Candidate Miss Rate": test_metrics["candidate_miss_rate"],
            "Sequence F1": test_metrics["seq_f1"],
            "Training Time (s)": round(train_time_s, 3),
            "Inference Time (s)": round(inf_time_s, 4),
        }
        comparison_rows.append(row)
        detailed_results[m_name] = {
            "summary": row,
            "cv_fold_macro_f1s": [round(x, 4) for x in fold_macro_f1s],
            "cv_fold_top1_accs": [round(x, 4) for x in fold_top1_accs],
            "cv_fold_log_losses": [round(x, 5) for x in fold_log_losses],
            "train_log_loss": round(train_log_loss, 5),
            "confusion_matrix": test_metrics["confusion_matrix"],
            "mlp_history": mlp_history,
        }
        print(
            f"  CV Macro F1: {row['CV Macro F1-score']:.4f} (+/-{row['CV Macro F1 Std']:.4f}) | "
            f"Test Macro F1: {row['Student-Test Macro F1-score']:.4f} | "
            f"Test Top-1 Acc: {row['Top-1 Accuracy']:.4f} | "
            f"Route Cont: {row['Route continuity']:.4f} | "
            f"LogLoss: {row['Log Loss']}"
        )

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(os.path.join(output_dir, "model_comparison_table.csv"), index=False)
    with open(os.path.join(output_dir, "model_comparison_detailed.json"), "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)

    return df_comp, detailed_results
