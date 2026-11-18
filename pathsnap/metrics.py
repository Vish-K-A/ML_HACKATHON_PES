"""Evaluation metrics: candidate-level classification + route-level matching."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def classification_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y, pc, labels=[0, 1]),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def top1_predictions(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Highest-scoring candidate per (trip, point)."""
    d = df[["trip", "point", "arc", "label"]].copy()
    d["score"] = score
    idx = d.groupby(["trip", "point"], sort=False)["score"].idxmax()
    return d.loc[idx].sort_values(["trip", "point"]).reset_index(drop=True)


def top1_accuracy(top: pd.DataFrame, n_points: int | None = None) -> float:
    """Share of observations whose top candidate is the reference arc.

    If n_points is given, observations without any candidate row are counted
    as errors as well.
    """
    return top.label.sum() / (n_points if n_points else len(top))


def continuity(arcs_seq: np.ndarray, net) -> float:
    """Share of consecutive predictions that are the same arc or directly connected."""
    if len(arcs_seq) < 2:
        return 1.0
    a, b = arcs_seq[:-1], arcs_seq[1:]
    same = net.canon[a] == net.canon[b]
    conn = net.arcs[a, 1] == net.arcs[b, 0]
    return float(np.mean(same | conn))


def continuity_from_arcs(arcs_seq: np.ndarray, arcs_tbl: np.ndarray, canon: np.ndarray) -> float:
    if len(arcs_seq) < 2:
        return 1.0
    a, b = arcs_seq[:-1], arcs_seq[1:]
    return float(np.mean((canon[a] == canon[b]) | (arcs_tbl[a, 1] == arcs_tbl[b, 0])))


def collapse(arcs_seq) -> list[int]:
    """Remove consecutive duplicates -> traversed arc sequence."""
    out = []
    for a in arcs_seq:
        a = int(a)
        if not out or out[-1] != a:
            out.append(a)
    return out


def route_overlap(pred_arcs, ref_arcs, length: np.ndarray, canon: np.ndarray) -> dict:
    """Length-weighted precision / recall / F1 of the traversed arc set."""
    P = set(canon[np.asarray(list(pred_arcs), dtype=int)].tolist()) if len(pred_arcs) else set()
    R = set(canon[np.asarray(list(ref_arcs), dtype=int)].tolist())
    inter = P & R
    lp = sum(length[a] for a in P) or 1e-9
    lr = sum(length[a] for a in R) or 1e-9
    li = sum(length[a] for a in inter)
    prec, rec = li / lp, li / lr
    return {"route_precision": prec, "route_recall": rec,
            "route_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
