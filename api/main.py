"""PathSnap FastAPI backend (runs on Laptop 1).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health            model + partition info
  GET  /trips             the evaluation trips the UI can offer (no labels)
  POST /predict           {"trip_id": "..."} -> full pipeline on that trip:
                          candidates -> features -> classifier -> HMM/Viterbi ->
                          route; then server-side scoring against the reference
                          route. The reference route itself is never returned.

Every request is logged to the console and to results/api_requests.log.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pathsnap.config import DATA_DIR, RESULTS_DIR, ROOT
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import evaluate_prediction, load_artifact, predict_trip


def _eval_partition() -> str:
    p = os.environ.get("PATHSNAP_EVAL_PARTITION")
    if p:
        return p
    return "evaluator_test" if (DATA_DIR / "evaluator_test").exists() else "student_test"


EVAL_PARTITION = _eval_partition()

log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(RESULTS_DIR / "api_requests.log", encoding="utf-8")):
    h.setFormatter(_fmt); log.addHandler(h)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.on_event("startup")
def _startup():
    art = load_artifact()
    log.info("model loaded: %s | HMM %s | evaluation partition: %s (%d trips) | data dir: %s",
             art["model_name"], art["hmm_params"], EVAL_PARTITION, len(list_trips(EVAL_PARTITION)), DATA_DIR)


@app.middleware("http")
async def _access_log(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code, (time.perf_counter() - t0) * 1000)
    return resp


@app.get("/health")
def health():
    art = load_artifact()
    return {"status": "ok", "model": art["model_name"], "hmm_params": art["hmm_params"],
            "evaluation_partition": EVAL_PARTITION, "n_trips": len(list_trips(EVAL_PARTITION))}


@app.get("/trips")
def trips():
    out = []
    for tid in list_trips(EVAL_PARTITION):
        tr = load_trip(EVAL_PARTITION, tid).track
        out.append({"trip_id": tid, "n_points": int(len(tr)),
                    "duration_min": round(float(tr[-1, 2] - tr[0, 2]) / 60, 1),
                    "center": [float(np.median(tr[:, 1])), float(np.median(tr[:, 0]))]})
    return {"partition": EVAL_PARTITION, "trips": out}


def _route_lines(trip, arc_seq) -> list[list[list[float]]]:
    n, lines, cur, last = trip.nodes, [], [], None
    for a in arc_seq:
        f, t = trip.arcs[int(a)]
        if last is not None and f == last:
            cur.append([round(float(n[t, 1]), 6), round(float(n[t, 0]), 6)])
        else:
            if cur:
                lines.append(cur)
            cur = [[round(float(n[f, 1]), 6), round(float(n[f, 0]), 6)], [round(float(n[t, 1]), 6), round(float(n[t, 0]), 6)]]
        last = t
    if cur:
        lines.append(cur)
    return lines


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in list_trips(EVAL_PARTITION):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.perf_counter()
    trip = load_trip(EVAL_PARTITION, req.trip_id)       # track + network only
    pred = predict_trip(trip)
    predict_s = time.perf_counter() - t0
    ev = evaluate_prediction(pred, EVAL_PARTITION, req.trip_id)  # server-side only
    ctx = pred["ctx"]
    from pathsnap.hmm import reconstruct_route
    metrics = {k: v for k, v in ev.items() if not k.startswith("_")}
    log.info("PREDICT trip=%s points=%d hmm_acc=%.4f clf_acc=%.4f route_f1=%.4f breaks=%d predict=%.2fs",
             req.trip_id, metrics["n_points"], metrics["hmm_accuracy"], metrics["classifier_top1_accuracy"],
             metrics["hmm_route_f1"], pred["hmm_info"]["n_breaks"], predict_s)
    correct = ev["_point_correct"]
    return {
        "trip_id": req.trip_id,
        "gps": [[round(float(la), 6), round(float(lo), 6), int(ok)] for lo, la, ok in
                zip(ctx.track[:, 0], ctx.track[:, 1], correct)],
        "predicted_route": _route_lines(trip, pred["route"]),
        "classifier_route": _route_lines(trip, reconstruct_route(ctx.net, pred["classifier_arcs"])),
        "metrics": metrics,
        "hmm_breaks": pred["hmm_info"]["n_breaks"],
        "timings_s": {**{k: round(v, 3) for k, v in pred["timings"].items()}, "total_predict": round(predict_s, 3)},
    }


@app.get("/")
def ui():
    """Convenience: serve the UI from the API host too (for single-laptop testing)."""
    return FileResponse(ROOT / "ui" / "index.html")
