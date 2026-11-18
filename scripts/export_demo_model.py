"""Interim demo artifact: untuned LightGBM -> models/pathsnap_final.joblib.

Usage: python scripts/export_demo_model.py

Fits the untuned LightGBM configuration from 04_bayes_opt.py on all 80 training
trips (observations whose reference arc is in the candidate set, same rows as
04/05) and exports it so the API can serve a real model while the Bayesian
search is still running. HMM parameters are copied from the existing artifact.
05_hmm_experiments.py later overwrites this file with the tuned model.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from pathsnap import config
from pathsnap.config import HMM_DEFAULTS, MODELS_DIR, RESULTS_DIR, SEED
from pathsnap.features import FEATURES

UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8)


def main():
    final = MODELS_DIR / "pathsnap_final.joblib"
    hmm_params = dict(HMM_DEFAULTS)
    if final.exists():
        old = joblib.load(final)
        hmm_params = old.get("hmm_params", hmm_params)
        backup = MODELS_DIR / f"pathsnap_final_{old.get('model_name', 'old')}.joblib.bak"
        shutil.copy2(final, backup)
        print("backed up previous artifact ->", backup.name)

    tr = pd.read_parquet(RESULTS_DIR / "features_train.parquet")
    pts = pd.read_parquet(RESULTS_DIR / "points_train.parquet")
    trf = tr.merge(pts.loc[pts.cand_hit == 1, ["trip", "point"]], on=["trip", "point"])
    print(f"fitting LightGBM (untuned) on {len(trf):,} candidate rows ...", flush=True)
    t0 = time.time()
    model = LGBMClassifier(**UNTUNED, random_state=SEED, verbose=-1, n_jobs=-1)
    model.fit(trf[FEATURES].to_numpy(), trf.label.to_numpy())
    print(f"fit done in {time.time() - t0:.1f}s")

    artifact = {"model": model, "model_name": "LightGBM-untuned", "model_params": UNTUNED,
                "features": FEATURES, "hmm_params": hmm_params,
                "candidate_config": {"radius_m": config.CAND_RADIUS_M, "k": config.CAND_K,
                                     "densify_step_m": config.DENSIFY_STEP_M, "margin_m": config.NETWORK_MARGIN_M},
                "trained_on": "train partition (80 trips), observations with reference arc in candidate set",
                "note": "interim demo artifact; replaced by 05_hmm_experiments.py with the tuned model",
                "seed": SEED, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    joblib.dump(artifact, final, compress=3)
    (MODELS_DIR / "pathsnap_final_config.json").write_text(json.dumps(
        {k: v for k, v in artifact.items() if k != "model"}, indent=2, default=str))
    print("exported", final)


if __name__ == "__main__":
    main()
