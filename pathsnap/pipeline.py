"""End-to-end inference pipeline used by the API and the final experiments.

predict_trip() only receives the trip's track and road network; the reference
route is never passed in. evaluate_prediction() is a separate, server-side step
that loads the reference route to score a finished prediction.
"""
from __future__ import annotations

import time
from functools import lru_cache

import joblib
import numpy as np

from .config import MODELS_DIR
from .features import FEATURES, build_context, build_features
from .hmm import reconstruct_route, viterbi
from .io import Trip, load_route
from .labels import align_route
from .metrics import collapse, continuity, route_overlap

ARTIFACT = MODELS_DIR / "pathsnap_final.joblib"


@lru_cache(maxsize=1)
def load_artifact(path: str = str(ARTIFACT)) -> dict:
    art = joblib.load(path)
    assert art["features"] == FEATURES, "feature list mismatch between artifact and code"
    return art


def predict_trip(trip: Trip, artifact: dict | None = None) -> dict:
    art = artifact or load_artifact()
    timings = {}
    t0 = time.perf_counter()
    ctx = build_context(trip)
    timings["candidates_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    df = build_features(ctx)
    timings["features_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = art["model"].predict_proba(df[FEATURES].to_numpy())[:, 1]
    timings["classifier_s"] = time.perf_counter() - t0
    sizes = [len(c[0]) for c in ctx.cands]
    per_point = np.split(p, np.cumsum(sizes)[:-1])
    clf_arcs = np.array([c[0][np.argmax(s)] if len(s) else -1 for c, s in zip(ctx.cands, per_point)])
    t0 = time.perf_counter()
    _, hmm_arcs, info = viterbi(ctx, per_point, art["hmm_params"])
    timings["viterbi_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    route = reconstruct_route(ctx.net, hmm_arcs)
    timings["route_fill_s"] = time.perf_counter() - t0
    return {"ctx": ctx, "scores": per_point, "classifier_arcs": clf_arcs, "hmm_arcs": hmm_arcs,
            "route": route, "hmm_info": info, "timings": timings}


def evaluate_prediction(pred: dict, partition: str, trip_id: str) -> dict:
    """Server-side scoring against the reference route (never sent to the model)."""
    ctx = pred["ctx"]
    net = ctx.net
    ref_route = load_route(partition, trip_id)
    ref_arc, ref_d = align_route(ctx.xy, ref_route, net)
    c = net.canon
    hit = np.array([np.isin(c[cc[0]], c[r]).any() for cc, r in zip(ctx.cands, ref_arc)])
    clf_ok = c[pred["classifier_arcs"]] == c[ref_arc]
    hmm_ok = c[pred["hmm_arcs"]] == c[ref_arc]
    ro_hmm = route_overlap(pred["route"], ref_route, net.length, c)
    ro_clf = route_overlap(reconstruct_route(net, pred["classifier_arcs"]), ref_route, net.length, c)
    return {
        "n_points": int(len(ref_arc)),
        "candidate_miss_rate": float(1 - hit.mean()),
        "classifier_top1_accuracy": float(clf_ok.mean()),
        "hmm_accuracy": float(hmm_ok.mean()),
        "classifier_continuity": continuity(pred["classifier_arcs"], net),
        "hmm_continuity": continuity(pred["hmm_arcs"], net),
        "classifier_route_f1": ro_clf["route_f1"],
        "hmm_route_precision": ro_hmm["route_precision"],
        "hmm_route_recall": ro_hmm["route_recall"],
        "hmm_route_f1": ro_hmm["route_f1"],
        "n_reference_arcs": int(len(ref_route)),
        "n_predicted_route_arcs": int(len(pred["route"])),
        "classifier_segments": int(len(collapse(pred["classifier_arcs"]))),
        "hmm_segments": int(len(collapse(pred["hmm_arcs"]))),
        "_ref_route": ref_route, "_point_correct": hmm_ok,
    }
