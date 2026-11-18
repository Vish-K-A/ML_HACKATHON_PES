"""Exploratory data analysis on the train and student_test partitions.

Outputs (results/eda/):
  inventory.csv            one row per trip: files, sizes, counts, gaps, extent,
                           network size/connectivity, route checks, alignment
  partition_summary.csv    aggregated statistics per partition
  data_quality_notes.md    issues found and decisions taken
  *.png                    plots
results/maps/eda_trip_<id>.html  Folium map(s): GPS, network, reference route
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.config import RESULTS_DIR
from pathsnap.features import build_context
from pathsnap.geo import haversine_m
from pathsnap.io import list_trips, load_route, load_trip, trip_dir
from pathsnap.labels import align_route
from pathsnap.viz import trip_map

OUT = RESULTS_DIR / "eda"
OUT.mkdir(parents=True, exist_ok=True)
PARTS = ["train", "student_test"]   # evaluator partition is never explored


def trip_row(part: str, tid: str) -> tuple[dict, np.ndarray]:
    d = trip_dir(part, tid)
    files = {ext: d / f"{tid}.{ext}" for ext in ("track", "nodes", "arcs", "route")}
    row = {"partition": part, "trip": tid}
    for ext, f in files.items():
        row[f"has_{ext}"] = f.exists()
        row[f"kb_{ext}"] = round(f.stat().st_size / 1024, 1) if f.exists() else np.nan
    trip = load_trip(part, tid)
    tr, nd, ar = trip.track, trip.nodes, trip.arcs
    route = load_route(part, tid)
    row.update(n_gps=len(tr), n_nodes=len(nd), n_arcs=len(ar), n_route=len(route),
               n_route_unique=len(np.unique(route)))
    row["n_nan_gps"] = int((~np.isfinite(tr)).any(1).sum())
    dt = np.diff(tr[:, 2])
    row.update(dt_min=dt.min(), dt_median=np.median(dt), dt_max=dt.max(),
               pct_dt_1s=np.mean(np.isclose(dt, 1.0)) * 100,
               n_gaps_gt5s=int((dt > 5).sum()), n_gaps_gt30s=int((dt > 30).sum()),
               n_nonincreasing_t=int((dt <= 0).sum()),
               duration_min=(tr[-1, 2] - tr[0, 2]) / 60)
    step = haversine_m(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    spd = step / np.maximum(dt, 1e-6)
    row.update(track_km=step.sum() / 1000, speed_median_mps=np.median(spd),
               speed_p99_mps=np.percentile(spd, 99), n_speed_gt70mps=int((spd > 70).sum()),
               lon_min=tr[:, 0].min(), lon_max=tr[:, 0].max(), lat_min=tr[:, 1].min(), lat_max=tr[:, 1].max(),
               lon_c=np.median(tr[:, 0]), lat_c=np.median(tr[:, 1]))
    # network
    deg = np.bincount(ar.ravel(), minlength=len(nd))
    row["n_isolated_nodes"] = int((deg == 0).sum())
    row["n_self_loops"] = int((ar[:, 0] == ar[:, 1]).sum())
    row["n_duplicate_arcs"] = int(len(ar) - len(np.unique(ar, axis=0)))
    g = coo_matrix((np.ones(len(ar)), (ar[:, 0], ar[:, 1])), shape=(len(nd), len(nd)))
    nw, lw = connected_components(g, directed=True, connection="weak")
    ns, ls = connected_components(g, directed=True, connection="strong")
    used = deg > 0
    row["n_weak_components"] = int(len(np.unique(lw[used])))
    row["largest_weak_comp_frac"] = np.bincount(lw[used]).max() / used.sum()
    row["largest_strong_comp_frac"] = np.bincount(ls[used]).max() / used.sum()
    key = ar[:, 0] * len(nd) + ar[:, 1]
    rkey = ar[:, 1] * len(nd) + ar[:, 0]
    row["pct_two_way_arcs"] = np.isin(rkey, key).mean() * 100
    # route
    row["route_ids_valid"] = bool(((route >= 0) & (route < len(ar))).all())
    row["route_pct_connected"] = np.mean(ar[route[:-1], 1] == ar[route[1:], 0]) * 100 if len(route) > 1 else np.nan
    row["route_repeats"] = int(len(route) - len(np.unique(route)))
    # geometry: GPS->route alignment (uses projection + candidate generation)
    ctx = build_context(trip)
    row["utm_epsg"] = ctx.net.proj.epsg
    lab, ld = align_route(ctx.xy, route, ctx.net)
    row.update(align_dist_median_m=np.median(ld), align_dist_p95_m=np.percentile(ld, 95),
               align_dist_max_m=ld.max(), pct_gps_gt60m_from_route=np.mean(ld > 60) * 100)
    hit = np.array([np.isin(ctx.net.canon[c[0]], ctx.net.canon[l]).any() for c, l in zip(ctx.cands, lab)])
    row["candidate_miss_pct"] = (1 - hit.mean()) * 100
    row["cands_per_point"] = np.mean([len(c[0]) for c in ctx.cands])
    row["route_km"] = ctx.net.length[route].sum() / 1000
    return row, dt


def main():
    rows, all_dt = [], {p: [] for p in PARTS}
    for part in PARTS:
        for tid in list_trips(part):
            r, dt = trip_row(part, tid)
            rows.append(r); all_dt[part].append(dt)
            print(part, tid, r["n_gps"], f"miss={r['candidate_miss_pct']:.2f}%")
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)

    num = ["n_gps", "duration_min", "track_km", "route_km", "n_nodes", "n_arcs", "n_route", "dt_max",
           "pct_dt_1s", "n_gaps_gt5s", "speed_median_mps", "align_dist_median_m", "align_dist_p95_m",
           "pct_gps_gt60m_from_route", "candidate_miss_pct", "cands_per_point", "route_pct_connected",
           "n_isolated_nodes", "n_duplicate_arcs", "largest_weak_comp_frac"]
    summ = inv.groupby("partition")[num].agg(["count", "mean", "median", "min", "max"]).T
    summ.to_csv(OUT / "partition_summary.csv")
    tot = inv.groupby("partition").agg(trips=("trip", "count"), gps_points=("n_gps", "sum"),
                                        route_arcs=("n_route", "sum"), nan_rows=("n_nan_gps", "sum"))
    tot.to_csv(OUT / "partition_totals.csv")
    print(tot)

    # ---------------- plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p in PARTS:
        ax[0].hist(inv.loc[inv.partition == p, "n_gps"], bins=25, alpha=.6, label=p)
        ax[1].hist(inv.loc[inv.partition == p, "track_km"], bins=25, alpha=.6, label=p)
        ax[2].hist(np.log10(inv.loc[inv.partition == p, "n_arcs"]), bins=25, alpha=.6, label=p)
    ax[0].set(title="GPS observations per trip", xlabel="points")
    ax[1].set(title="Track length per trip", xlabel="km")
    ax[2].set(title="Road-network size", xlabel="log10(arcs)")
    for a in ax: a.legend()
    fig.tight_layout(); fig.savefig(OUT / "trip_size_distributions.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    dt = np.concatenate(all_dt["train"])
    vals, cnt = np.unique(np.round(dt), return_counts=True)
    ax[0].bar(vals[vals <= 10].astype(int).astype(str), cnt[vals <= 10])
    ax[0].set_yscale("log"); ax[0].set(title="Sampling interval (train)", xlabel="seconds (rounded)", ylabel="count")
    big = dt[dt > 5]
    ax[1].hist(big, bins=40); ax[1].set(title=f"Gaps > 5 s (train, n={len(big)})", xlabel="seconds")
    fig.tight_layout(); fig.savefig(OUT / "sampling_intervals.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for p, m in zip(PARTS, ["o", "s"]):
        s = inv[inv.partition == p]
        ax.scatter(s.lon_c, s.lat_c, marker=m, label=p, alpha=.7)
    ax.set(xlim=(-180, 180), ylim=(-60, 80), xlabel="longitude", ylabel="latitude",
           title="Trip locations (track median)")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "trip_locations.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(inv.align_dist_median_m, inv.align_dist_p95_m, c=(inv.partition == "train"), cmap="coolwarm")
    ax[0].set(xscale="log", yscale="log", xlabel="median GPS-to-reference distance (m)",
              ylabel="95th percentile (m)", title="GPS noise vs reference route")
    ax[1].bar(range(len(inv)), inv.sort_values("candidate_miss_pct").candidate_miss_pct)
    ax[1].set(title="Candidate miss rate per trip (sorted)", ylabel="% observations", xlabel="trip")
    fig.tight_layout(); fig.savefig(OUT / "noise_and_candidate_miss.png", dpi=130); plt.close(fig)

    # missing-value analysis
    miss = inv[[c for c in inv.columns if c.startswith("has_")]].apply(lambda c: (~c).sum())
    miss["gps_rows_with_nan"] = inv.n_nan_gps.sum()
    miss.to_frame("missing_count").to_csv(OUT / "missing_values.csv")

    # one representative example GPS trajectory panel
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for a, (_, r) in zip(axs.ravel(), inv[inv.partition == "train"].sample(8, random_state=0).iterrows()):
        t = load_trip("train", r.trip)
        a.plot(t.track[:, 0], t.track[:, 1], ".", ms=1.5)
        a.set_title(f"{r.trip} ({r.n_gps} pts)", fontsize=9); a.set_aspect(1 / np.cos(np.radians(r.lat_c)))
        a.tick_params(labelsize=6)
    fig.suptitle("Raw GPS trajectories (train, random sample)")
    fig.tight_layout(); fig.savefig(OUT / "trajectory_samples.png", dpi=130); plt.close(fig)

    # road network sample plot
    t = load_trip("train", "00000001"); ctx = build_context(t)
    sel = ctx.net.indexed_arcs
    fig, ax = plt.subplots(figsize=(8, 8))
    segs = np.stack([ctx.net.A[sel], ctx.net.B[sel]], 1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors="0.6", linewidths=.4))
    r = load_route("train", "00000001")
    ax.add_collection(LineCollection(np.stack([ctx.net.A[r], ctx.net.B[r]], 1), colors="green", linewidths=2))
    ax.plot(ctx.xy[:, 0], ctx.xy[:, 1], ".", ms=2, color="tab:blue")
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title("Trip 00000001: clipped road network (grey), reference route (green), GPS (blue)")
    fig.tight_layout(); fig.savefig(OUT / "road_network_trip00000001.png", dpi=130); plt.close(fig)

    # folium maps for two training trips
    for tid in ["00000001", inv[inv.partition == "train"].sort_values("pct_gps_gt60m_from_route").trip.iloc[-1]]:
        m = trip_map(load_trip("train", tid), reference=load_route("train", tid))
        m.save(str(RESULTS_DIR / "maps" / f"eda_trip_{tid}.html"))

    write_notes(inv)


def write_notes(inv: pd.DataFrame):
    tr = inv[inv.partition == "train"]
    worst = inv.sort_values("pct_gps_gt60m_from_route", ascending=False).head(5)
    lines = [
        "# Data-quality notes (generated by scripts/01_eda.py)\n",
        f"* Partitions explored: train ({(inv.partition=='train').sum()} trips), student_test "
        f"({(inv.partition=='student_test').sum()} trips). Evaluator partition not explored.",
        f"* All four files present for every trip: {bool(inv[[c for c in inv if c.startswith('has_')]].all().all())}.",
        f"* GPS rows with NaN: {int(inv.n_nan_gps.sum())}; non-increasing timestamps: {int(inv.n_nonincreasing_t.sum())}.",
        f"* Sampling: median {inv.dt_median.median():.0f} s; share of exact 1 s steps "
        f"{inv.pct_dt_1s.mean():.1f}% (mean over trips); max gap {inv.dt_max.max():.0f} s; "
        f"trips with a gap > 30 s: {(inv.n_gaps_gt30s>0).sum()}.",
        f"* Trips span {inv.utm_epsg.nunique()} UTM zones on several continents, so each trip is projected to its own UTM zone.",
        f"* Road networks: {inv.n_arcs.min()}–{inv.n_arcs.max()} arcs; duplicate arcs in "
        f"{(inv.n_duplicate_arcs>0).sum()} trips (mapped to a canonical arc); self-loops in "
        f"{(inv.n_self_loops>0).sum()} trips; isolated nodes in {(inv.n_isolated_nodes>0).sum()} trips.",
        f"* Two-way roads are stored as two opposite arcs (mean {inv.pct_two_way_arcs.mean():.0f}% of arcs have a reverse twin), "
        "so direction of travel matters: heading features are required.",
        f"* Reference routes: ids valid in all trips = {bool(inv.route_ids_valid.all())}; "
        f"consecutive arcs connected {inv.route_pct_connected.mean():.2f}% on average; "
        f"trips with repeated arcs (loops / out-and-back): {(inv.route_repeats>0).sum()}.",
        "* The .route file is one arc sequence per trip, not one label per GPS row. Per-observation labels are "
        "obtained with a monotone alignment (pathsnap/labels.py).",
        f"* Median GPS-to-reference distance {inv.align_dist_median_m.median():.1f} m; but some trips contain GPS "
        "sections far from the reference route (reference may not cover the full track, or GPS drift):",
    ]
    for _, r in worst.iterrows():
        lines.append(f"  * {r.partition}/{r.trip}: {r.pct_gps_gt60m_from_route:.1f}% of points > 60 m from route, max {r.align_dist_max_m:.0f} m")
    lines += [
        f"* Candidate generation (radius 60 m, max 12 arcs): mean miss rate {inv.candidate_miss_pct.mean():.2f}% "
        f"(train {tr.candidate_miss_pct.mean():.2f}%), {inv.cands_per_point.mean():.1f} candidates per point on average.",
        "* Decisions: invalid/duplicate-time rows removed; points whose reference arc is outside the candidate set "
        "are kept and always count as errors in Top-1 accuracy (no optimistic filtering).",
    ]
    (OUT / "data_quality_notes.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
