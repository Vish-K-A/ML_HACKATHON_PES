"""
FastAPI Backend for PathSnap (Laptop 1: Model, HMM/Viterbi Decoder, and Server-Side Evaluation).
Strictly isolates evaluator ground-truth (.route) on the server side so it never enters
the prediction pipeline or UI.
"""

import datetime
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import folium
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_processing.preprocess import align_reference_route, load_trip
from feature_engineering.features import extract_trip_candidates_and_features
from hmm_viterbi.viterbi_decoder import HMMConfig, ViterbiRouteDecoder
from model_evaluation.metrics import (
    compute_reconstructed_route_continuity,
    compute_route_continuity,
    compute_sequence_overlap_metrics,
)
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

app = FastAPI(
    title="PathSnap Distributed Map-Matching & Route Reconstruction API",
    description="Laptop 1 Backend: Hosts the Bayesian-tuned LightGBM classifier, HMM/Viterbi decoder, and server-side evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("PATHSNAP_MODEL_PATH", os.path.join(ROOT_DIR, "models", "final_model.joblib"))
STUDENT_FOLDER = os.environ.get("PATHSNAP_DATA_DIR", os.path.join(ROOT_DIR, "Student folder"))

_ARTIFACT: Optional[Dict[str, Any]] = None


def get_model_artifact() -> Dict[str, Any]:
    global _ARTIFACT
    if _ARTIFACT is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Trained model artifact not found at {MODEL_PATH}. Run `python run_pipeline.py` first.")
        _ARTIFACT = joblib.load(MODEL_PATH)
        print(f"[PathSnap API] Loaded final model artifact: {_ARTIFACT.get('model_name')} from {MODEL_PATH}")
    return _ARTIFACT


def discover_evaluation_trips() -> Dict[str, Tuple[str, str]]:
    """
    Discovers available evaluation trips. Prioritizes `evaluator_test` if present,
    alongside `student_test` (the 10 reserved testing trips).
    Returns mapping: trip_id -> (partition_name, trip_dir_path)
    """
    trips: Dict[str, Tuple[str, str]] = {}
    for part in ["evaluator_test", "student_test"]:
        part_dir = os.path.join(STUDENT_FOLDER, part)
        if os.path.isdir(part_dir):
            for tid in sorted(os.listdir(part_dir)):
                tdir = os.path.join(part_dir, tid)
                if os.path.isdir(tdir) and os.path.exists(os.path.join(tdir, f"{tid}.track")):
                    trips[tid] = (part, tdir)
    return trips


class PredictRequest(BaseModel):
    trip_id: str


def build_prediction_folium_html(
    trip_id: str,
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    clf_pred_arcs: np.ndarray,
    hmm_pred_arcs: np.ndarray,
    reconstructed_route: np.ndarray,
) -> str:
    """
    Builds an interactive Folium map HTML string for the UI without exposing raw ground-truth labels.
    """
    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    # Road network subset
    fg_net = folium.FeatureGroup(name="Road Network Geometry (.arcs)", show=True)
    lon_min, lon_max = track[:, 0].min() - 0.0015, track[:, 0].max() + 0.0015
    lat_min, lat_max = track[:, 1].min() - 0.0015, track[:, 1].max() + 0.0015
    u_lonlat = nodes[arcs[:, 0]]
    v_lonlat = nodes[arcs[:, 1]]
    mask = (
        (u_lonlat[:, 0] >= lon_min) & (u_lonlat[:, 0] <= lon_max) &
        (u_lonlat[:, 1] >= lat_min) & (u_lonlat[:, 1] <= lat_max)
    )
    net_indices = np.where(mask)[0]
    step_net = max(1, len(net_indices) // 1000)
    for a_idx in net_indices[::step_net]:
        u_pt = u_lonlat[a_idx]
        v_pt = v_lonlat[a_idx]
        folium.PolyLine(
            locations=[[float(u_pt[1]), float(u_pt[0])], [float(v_pt[1]), float(v_pt[0])]],
            color="#888888", weight=1.5, opacity=0.45,
        ).add_to(fg_net)
    fg_net.add_to(fmap)

    # Classifier-only route
    fg_clf = folium.FeatureGroup(name="Classifier-Only Predictions (No HMM)", show=False)
    clf_seq = [int(clf_pred_arcs[0])]
    for a in clf_pred_arcs[1:]:
        if int(a) != clf_seq[-1]:
            clf_seq.append(int(a))
    for c_arc in clf_seq:
        u_pt = nodes[arcs[c_arc, 0]]
        v_pt = nodes[arcs[c_arc, 1]]
        folium.PolyLine(
            locations=[[float(u_pt[1]), float(u_pt[0])], [float(v_pt[1]), float(v_pt[0])]],
            color="#ff7f00", weight=4.0, opacity=0.80, dash_array="6, 6",
            tooltip=f"Classifier-Only Segment {c_arc}",
        ).add_to(fg_clf)
    fg_clf.add_to(fmap)

    # HMM + Viterbi Reconstructed Route
    fg_hmm = folium.FeatureGroup(name="Predicted Route (Tuned LightGBM + HMM/Viterbi)", show=True)
    for p_arc in reconstructed_route:
        u_pt = nodes[arcs[int(p_arc), 0]]
        v_pt = nodes[arcs[int(p_arc), 1]]
        folium.PolyLine(
            locations=[[float(u_pt[1]), float(u_pt[0])], [float(v_pt[1]), float(v_pt[0])]],
            color="#e41a1c", weight=4.5, opacity=0.92,
            tooltip=f"Predicted Route Arc {int(p_arc)}",
        ).add_to(fg_hmm)
    fg_hmm.add_to(fmap)

    # Raw GPS observations
    fg_gps = folium.FeatureGroup(name="Noisy GPS Observations (.track)", show=True)
    T = len(track)
    step_gps = max(1, T // 350)
    for t_i in range(0, T, step_gps):
        lon_i, lat_i, ts_i = track[t_i]
        folium.CircleMarker(
            location=[float(lat_i), float(lon_i)],
            radius=3.2,
            color="#0055cc",
            fill=True,
            fill_color="#3388ff",
            fill_opacity=0.85,
            tooltip=f"t={int(ts_i)}s | Predicted Arc={int(hmm_pred_arcs[t_i])}",
        ).add_to(fg_gps)

    folium.Marker(
        location=[float(track[0, 1]), float(track[0, 0])],
        popup=f"Start (Trip {trip_id})",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fg_gps)
    folium.Marker(
        location=[float(track[-1, 1]), float(track[-1, 0])],
        popup=f"End (Trip {trip_id})",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(fg_gps)
    fg_gps.add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    fmap.fit_bounds([[float(track[:, 1].min()), float(track[:, 0].min())], [float(track[:, 1].max()), float(track[:, 0].max())]])
    return fmap._repr_html_()


@app.get("/health")
def health_check(request: Request) -> Dict[str, Any]:
    artifact = get_model_artifact()
    trips = discover_evaluation_trips()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] [API /health] Request from {request.client.host if request.client else 'local'} | {len(trips)} evaluation trips ready.")
    return {
        "status": "online",
        "model_name": artifact.get("model_name"),
        "available_evaluation_trips": list(trips.keys()),
        "k_candidates": artifact.get("k_candidates", 8),
    }


@app.get("/trips")
def list_trips(request: Request) -> Dict[str, Any]:
    trips = discover_evaluation_trips()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] [API /trips] Client {request.client.host if request.client else 'local'} requested trip list -> returning {len(trips)} trips: {list(trips.keys())}")
    trip_details = []
    for tid, (part, tdir) in trips.items():
        track_file = os.path.join(tdir, f"{tid}.track")
        with open(track_file, "r", encoding="utf-8") as f:
            n_obs = sum(1 for line in f if line.strip())
        trip_details.append({"trip_id": tid, "partition": part, "n_gps_observations": n_obs})
    return {"count": len(trip_details), "trips": trip_details}


def _process_trip_inference(trip_id: str, client_host: str) -> Dict[str, Any]:
    trips = discover_evaluation_trips()
    if trip_id not in trips:
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found in evaluation partitions. Available: {list(trips.keys())}")

    partition, tdir = trips[trip_id]
    artifact = get_model_artifact()
    clf = artifact["classifier"]
    k_cands = int(artifact.get("k_candidates", 8))
    decoder = ViterbiRouteDecoder(HMMConfig(**artifact["hmm_config"]))

    t_start = time.perf_counter()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 78)
    print(f"[{now_str}] [INFERENCE REQUEST] Client={client_host} | Selected Trip ID={trip_id} ({partition})")

    # 1. Load trip strictly WITHOUT ground-truth route for the prediction pipeline
    trip_blind = load_trip(tdir, trip_id=trip_id, partition=partition, load_route=False)
    feats = extract_trip_candidates_and_features(trip_blind, k_candidates=k_cands, include_labels=False)
    X = feats["X"]  # (T, K, F)
    T, K, F = X.shape

    # 2. Classifier candidate probability inference
    t_clf0 = time.perf_counter()
    probs = clf.predict_proba(X.reshape(T * K, F))[:, 1].reshape(T, K)
    clf_time_s = time.perf_counter() - t_clf0

    clf_top1_idx = np.argmax(probs, axis=1)
    clf_pred_arcs = feats["cand_arcs"][np.arange(T), clf_top1_idx]

    # 3. HMM + Viterbi Route Reconstruction
    t_hmm0 = time.perf_counter()
    dec_out = decoder.decode(
        arcs=trip_blind.arcs,
        nodes_xy=trip_blind.nodes_xy,
        cand_arcs=feats["cand_arcs"],
        probs=probs,
        perp_dist=feats["perp_dist"],
        proj_frac=feats["proj_frac"],
        cos_blend=feats["cos_blend"],
        dt_prev=feats["dt_prev"],
    )
    hmm_time_s = time.perf_counter() - t_hmm0
    hmm_pred_arcs = dec_out["obs_pred_arcs"]
    reconstructed_route = dec_out["reconstructed_route"]

    # 4. Server-Side Evaluation using internal reference data (never exposed to model or UI)
    eval_metrics: Dict[str, Any] = {
        "clf_route_continuity": round(compute_route_continuity(trip_blind.arcs, clf_pred_arcs), 4),
        "hmm_route_continuity": round(compute_route_continuity(trip_blind.arcs, hmm_pred_arcs), 4),
        "reconstructed_strict_continuity": round(compute_reconstructed_route_continuity(trip_blind.arcs, reconstructed_route), 4),
    }

    route_file = os.path.join(tdir, f"{trip_id}.route")
    if os.path.exists(route_file):
        trip_ref = load_trip(tdir, trip_id=trip_id, partition=partition, load_route=True)
        gt_arcs = align_reference_route(trip_ref)
        y_true = (feats["cand_arcs"] == gt_arcs[:, None]).astype(np.int32)
        y_flat = y_true.reshape(-1)
        p_flat = probs.reshape(-1)
        pred_bin = (p_flat >= 0.5).astype(np.int32)

        route_set = set(map(int, trip_ref.route))
        clf_top1_acc = float(np.mean(clf_pred_arcs == gt_arcs))
        hmm_top1_acc = float(np.mean(hmm_pred_arcs == gt_arcs))
        clf_route_match = float(np.mean([int(a) in route_set for a in clf_pred_arcs]))
        hmm_route_match = float(np.mean([int(a) in route_set for a in hmm_pred_arcs]))
        seq_ov = compute_sequence_overlap_metrics(reconstructed_route, trip_ref.route)

        eval_metrics.update({
            "classifier_top1_accuracy": round(clf_top1_acc, 4),
            "hmm_viterbi_top1_accuracy": round(hmm_top1_acc, 4),
            "classifier_route_match_accuracy": round(clf_route_match, 4),
            "hmm_viterbi_route_match_accuracy": round(hmm_route_match, 4),
            "candidate_macro_f1": round(float(f1_score(y_flat, pred_bin, average="macro", zero_division=0)), 4),
            "candidate_precision": round(float(precision_score(y_flat, pred_bin, average="macro", zero_division=0)), 4),
            "candidate_recall": round(float(recall_score(y_flat, pred_bin, average="macro", zero_division=0)), 4),
            "candidate_log_loss": round(float(log_loss(y_flat, np.clip(p_flat, 1e-6, 1.0 - 1e-6))), 5),
            "candidate_miss_rate": round(float(np.mean(np.sum(y_true, axis=1) == 0)), 4),
            "sequence_f1": seq_ov["seq_f1"],
            "sequence_jaccard": seq_ov["seq_jaccard"],
        })

    map_html = build_prediction_folium_html(
        trip_id=trip_id,
        track=trip_blind.track,
        nodes=trip_blind.nodes,
        arcs=trip_blind.arcs,
        clf_pred_arcs=clf_pred_arcs,
        hmm_pred_arcs=hmm_pred_arcs,
        reconstructed_route=reconstructed_route,
    )

    total_time_s = time.perf_counter() - t_start
    print(
        f"[PathSnap API] Completed Trip {trip_id}: T={T} GPS pts | "
        f"Classifier Top-1={eval_metrics.get('classifier_top1_accuracy', 'N/A')} -> "
        f"HMM+Viterbi Top-1={eval_metrics.get('hmm_viterbi_top1_accuracy', 'N/A')} | "
        f"Route Match={eval_metrics.get('hmm_viterbi_route_match_accuracy', 'N/A')} | "
        f"Continuity={eval_metrics.get('hmm_route_continuity')} | "
        f"Total Time={total_time_s:.3f}s"
    )
    print("=" * 78 + "\n")

    return {
        "trip_id": trip_id,
        "partition": partition,
        "model_used": artifact.get("model_name"),
        "n_gps_observations": int(T),
        "n_road_nodes": int(len(trip_blind.nodes)),
        "n_road_segments": int(len(trip_blind.arcs)),
        "n_reconstructed_route_segments": int(len(reconstructed_route)),
        "timing": {
            "classifier_inference_s": round(clf_time_s, 4),
            "hmm_viterbi_decode_s": round(hmm_time_s, 4),
            "total_pipeline_s": round(total_time_s, 4),
        },
        "evaluation_metrics": eval_metrics,
        "predicted_observation_arcs": hmm_pred_arcs.tolist(),
        "reconstructed_route_arcs": reconstructed_route.tolist(),
        "folium_map_html": map_html,
    }


@app.get("/predict/{trip_id}")
def predict_trip_get(trip_id: str, request: Request) -> Dict[str, Any]:
    client_host = request.client.host if request.client else "local"
    return _process_trip_inference(trip_id, client_host)


@app.post("/predict")
def predict_trip_post(req: PredictRequest, request: Request) -> Dict[str, Any]:
    client_host = request.client.host if request.client else "local"
    return _process_trip_inference(req.trip_id, client_host)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
