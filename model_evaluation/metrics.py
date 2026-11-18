"""
Evaluation Metrics for PathSnap Candidate Classification and Sequential Route Reconstruction.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def compute_route_continuity(arcs: np.ndarray, obs_pred_arcs: np.ndarray) -> float:
    """
    Computes observation-level route continuity across consecutive GPS observations (t-1, t):
    Fraction of steps where predicted segment at t is identical to t-1, directly adjacent
    (v_{t-1} == u_t and not an immediate U-turn), or connected via 1 intermediate segment.
    """
    T = len(obs_pred_arcs)
    if T <= 1:
        return 1.0

    a_prev = obs_pred_arcs[:-1]
    a_curr = obs_pred_arcs[1:]
    same = (a_prev == a_curr)

    u_prev = arcs[a_prev, 0]
    v_prev = arcs[a_prev, 1]
    u_curr = arcs[a_curr, 0]
    v_curr = arcs[a_curr, 1]
    not_uturn = (v_curr != u_prev)
    direct = (v_prev == u_curr) & not_uturn

    # Check 2-hop only for remaining non-same, non-direct transitions
    rem_idx = np.where(~(same | direct))[0]
    hop2_hits = 0
    if len(rem_idx) > 0:
        needed_v = set(map(int, v_prev[rem_idx]))
        from collections import defaultdict
        out_nodes = defaultdict(set)
        mask_arcs = np.isin(arcs[:, 0], list(needed_v))
        for u, v in arcs[mask_arcs]:
            out_nodes[int(u)].add(int(v))
        for idx in rem_idx:
            if not_uturn[idx] and int(u_curr[idx]) in out_nodes.get(int(v_prev[idx]), ()):
                hop2_hits += 1

    valid_steps = int(np.sum(same | direct)) + hop2_hits
    return float(valid_steps) / float(T - 1)


def compute_reconstructed_route_continuity(arcs: np.ndarray, route_seq: np.ndarray) -> float:
    """
    Computes strict node-to-node continuity (v_i == u_{i+1}) on a deduplicated route sequence.
    """
    if len(route_seq) <= 1:
        return 1.0
    v_prev = arcs[route_seq[:-1], 1]
    u_next = arcs[route_seq[1:], 0]
    return float(np.mean(v_prev == u_next))


def compute_sequence_overlap_metrics(pred_route: np.ndarray, gt_route: np.ndarray) -> Dict[str, float]:
    """
    Computes sequence-level precision, recall, F1, and Jaccard overlap between
    the reconstructed route and reference .route.
    """
    pred_set = set(map(int, pred_route))
    gt_set = set(map(int, gt_route))
    if not pred_set and not gt_set:
        return {"seq_precision": 1.0, "seq_recall": 1.0, "seq_f1": 1.0, "seq_jaccard": 1.0}
    inter = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    prec = inter / max(1, len(pred_set))
    rec = inter / max(1, len(gt_set))
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    jaccard = inter / max(1, union)
    return {
        "seq_precision": round(float(prec), 4),
        "seq_recall": round(float(rec), 4),
        "seq_f1": round(float(f1), 4),
        "seq_jaccard": round(float(jaccard), 4),
    }


def evaluate_predictions_on_trips(
    trip_cache: List[Dict[str, Any]],
    probs_per_trip: List[np.ndarray],
    pred_labels_per_trip: Optional[List[np.ndarray]] = None,
    has_probabilities: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates a model across a list of trips on both candidate-level classification metrics
    and observation/route-level map-matching metrics.
    """
    all_y_true = []
    all_y_pred = []
    all_probs = []

    total_obs = 0
    correct_top1 = 0
    correct_in_route = 0
    candidate_misses = 0
    continuities = []
    seq_f1s = []
    seq_jaccards = []

    for idx, tdata in enumerate(trip_cache):
        y = tdata["y"]                  # (T, K)
        cand_arcs = tdata["cand_arcs"]  # (T, K)
        gt_arcs = tdata["gt_arcs"]      # (T,)
        arcs = tdata["arcs"]            # (A, 2)
        route = tdata["route"]          # (M,)
        scores = probs_per_trip[idx]    # (T, K)
        T, K = y.shape

        y_flat = y.reshape(-1)
        s_flat = scores.reshape(-1)
        if pred_labels_per_trip is not None:
            p_flat = pred_labels_per_trip[idx].reshape(-1)
        else:
            p_flat = (s_flat >= 0.5).astype(np.int32)

        all_y_true.append(y_flat)
        all_y_pred.append(p_flat)
        all_probs.append(s_flat)

        # Observation-level Top-1 selection
        top1_idx = np.argmax(scores, axis=1)
        pred_obs_arcs = cand_arcs[np.arange(T), top1_idx]

        total_obs += T
        correct_top1 += int(np.sum(pred_obs_arcs == gt_arcs))
        route_set = set(map(int, route))
        correct_in_route += int(sum(1 for a in pred_obs_arcs if int(a) in route_set))
        candidate_misses += int(np.sum(np.sum(y, axis=1) == 0))

        continuities.append(compute_route_continuity(arcs, pred_obs_arcs))
        seq_m = compute_sequence_overlap_metrics(pred_obs_arcs, route)
        seq_f1s.append(seq_m["seq_f1"])
        seq_jaccards.append(seq_m["seq_jaccard"])

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    probs_all = np.concatenate(all_probs)

    acc = float(accuracy_score(y_true_all, y_pred_all))
    prec_macro = float(precision_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    prec_pos = float(precision_score(y_true_all, y_pred_all, pos_label=1, zero_division=0))
    rec_pos = float(recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0))
    f1_pos = float(f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true_all, y_pred_all).tolist()

    ll = None
    if has_probabilities:
        clipped = np.clip(probs_all, 1e-6, 1.0 - 1e-6)
        ll = round(float(log_loss(y_true_all, clipped)), 5)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "pos_precision": round(prec_pos, 4),
        "pos_recall": round(rec_pos, 4),
        "pos_f1": round(f1_pos, 4),
        "log_loss": ll if ll is not None else "N/A",
        "top1_accuracy": round(correct_top1 / max(1, total_obs), 4),
        "route_match_accuracy": round(correct_in_route / max(1, total_obs), 4),
        "candidate_miss_rate": round(candidate_misses / max(1, total_obs), 4),
        "route_continuity": round(float(np.mean(continuities)), 4),
        "seq_f1": round(float(np.mean(seq_f1s)), 4),
        "seq_jaccard": round(float(np.mean(seq_jaccards)), 4),
        "confusion_matrix": cm,
    }
