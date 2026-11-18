"""Build the candidate-level feature tables for train and student_test.

Outputs:
  results/features_<partition>.parquet   one row per (observation, candidate)
  results/points_<partition>.parquet     one row per observation: reference arc,
                                         alignment distance, candidate hit flag
  results/eda/candidate_sensitivity.csv/png  miss rate vs radius / K (train only)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.config import CAND_K, CAND_RADIUS_M, RESULTS_DIR
from pathsnap.features import attach_labels, build_context, build_features
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.labels import align_route


def process(part: str, tid: str, sensitivity: bool):
    t0 = time.time()
    ctx = build_context(load_trip(part, tid))
    lab, ld = align_route(ctx.xy, load_route(part, tid), ctx.net)
    df = attach_labels(build_features(ctx), ctx, lab)
    hit = df.groupby("point").label.max().reindex(range(len(ctx.xy)), fill_value=0).to_numpy()
    pts = pd.DataFrame({"trip": tid, "point": np.arange(len(ctx.xy)), "ref_arc": lab,
                        "ref_canon": ctx.net.canon[lab], "align_dist": ld, "cand_hit": hit,
                        "t": ctx.track[:, 2]})
    sens = None
    if sensitivity:
        big = ctx.net.candidates(ctx.xy, radius=120.0, k=40)
        rows = []
        for i, (arcs, d, _) in enumerate(big):
            ok = ctx.net.canon[arcs] == ctx.net.canon[lab[i]]
            # rank / distance of reference arc inside the big candidate list
            rows.append((d[ok][0] if ok.any() else np.inf, int(np.argmax(ok)) if ok.any() else 999))
        sens = pd.DataFrame(rows, columns=["ref_dist", "ref_rank"]).assign(trip=tid)
    return df, pts, sens, time.time() - t0


def main():
    for part in ["train", "student_test"]:
        trips = list_trips(part)
        res = Parallel(n_jobs=8, verbose=5)(delayed(process)(part, t, part == "train") for t in trips)
        feats = pd.concat([r[0] for r in res], ignore_index=True)
        pts = pd.concat([r[1] for r in res], ignore_index=True)
        feats.to_parquet(RESULTS_DIR / f"features_{part}.parquet", index=False)
        pts.to_parquet(RESULTS_DIR / f"points_{part}.parquet", index=False)
        print(part, "rows", len(feats), "points", len(pts), "pos rate", feats.label.mean().round(4),
              "miss", (1 - pts.cand_hit.mean()).round(4), "build s", round(sum(r[3] for r in res), 1))
        if part == "train":
            s = pd.concat([r[2] for r in res], ignore_index=True)
            grid = []
            for R in [20, 30, 45, 60, 80, 100, 120]:
                for K in [4, 6, 8, 12, 16, 24, 40]:
                    miss = 1 - np.mean((s.ref_dist <= R) & (s.ref_rank < K))
                    grid.append((R, K, miss * 100))
            g = pd.DataFrame(grid, columns=["radius_m", "k", "miss_pct"])
            g.to_csv(RESULTS_DIR / "eda" / "candidate_sensitivity.csv", index=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            for K, gg in g.groupby("k"):
                ax.plot(gg.radius_m, gg.miss_pct, marker="o", label=f"K={K}")
            ax.axvline(CAND_RADIUS_M, ls="--", c="k", lw=.8)
            ax.set(xlabel="candidate radius (m)", ylabel="candidate miss rate (%)", yscale="log",
                   title=f"Candidate miss rate vs radius/K (train). Chosen: R={CAND_RADIUS_M:.0f}, K={CAND_K}")
            ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.3)
            fig.tight_layout(); fig.savefig(RESULTS_DIR / "eda" / "candidate_sensitivity.png", dpi=130)
            print(g.pivot(index="radius_m", columns="k", values="miss_pct").round(2))


if __name__ == "__main__":
    main()
