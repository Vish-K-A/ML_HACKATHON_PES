"""Central configuration for PathSnap.

Every tunable constant used by the pipeline lives here so that experiments,
the API and the report all refer to the same values.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data"))
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

PARTITIONS = ("train", "student_test", "evaluator_test")

SEED = 42

# ---------------------------------------------------------------- candidates
# A GPS point gets every arc whose segment lies within CAND_RADIUS_M, capped to
# the CAND_K nearest arcs. Values were chosen from the candidate-miss analysis
# in scripts/02_build_dataset.py (training partition only).
CAND_RADIUS_M = 60.0
CAND_K = 12
# arcs are densified into sample points at most this far apart for the KD-tree
DENSIFY_STEP_M = 15.0
# network is clipped to the track bounding box expanded by this margin
NETWORK_MARGIN_M = 1000.0

# ---------------------------------------------------------------- HMM
HMM_DEFAULTS = {
    "beta": 5.0,           # exponential scale (m) on |route_dist - gps_dist|
    "emission_power": 1.0, # weight on log classifier probability
    "max_speed_mps": 55.0, # route search cut-off speed
    "min_search_m": 300.0, # minimum route search radius
    "gap_break_s": 120.0,  # gaps above this break the chain (restart Viterbi)
}
