"""Candidate generation and feature engineering.

Each GPS observation i has a set of candidate arcs C_i. The learning problem is
binary classification of (observation, candidate) pairs: "is this candidate the
reference arc of this observation?". Classifier scores are later normalised
within C_i and used as HMM emission probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import Projector, angle_diff, bearing, point_segment
from .io import Trip, clean_track
from .network import RoadNetwork

FEATURES = [
    "dist", "log_dist", "rank", "dist_minus_min", "dist_ratio_min", "gauss_share",
    "n_cand", "t", "at_end", "arc_len", "log_arc_len",
    "head_diff", "cos_head", "head_diff_rev", "speed", "heading_reliable", "two_way",
    "dist_prev", "dist_next", "win_mean_dist", "win_max_dist", "win_rank_mean",
    "out_deg_to", "in_deg_from", "dt_prev", "dt_next",
]
WIN = 3            # +/- observations used for window features
GAUSS_SIGMA = 20.0  # metres, only used for the gauss_share feature


@dataclass
class TripContext:
    trip: Trip
    track: np.ndarray       # cleaned track (lon, lat, t)
    clean_report: dict
    xy: np.ndarray          # projected GPS points
    net: RoadNetwork
    cands: list             # per point (arcs, dist, t)


def build_context(trip: Trip) -> TripContext:
    track, rep = clean_track(trip.track)
    lon0, lat0 = np.median(track[:, 0]), np.median(track[:, 1])
    proj = Projector(lon0, lat0)
    xy = proj(track[:, :2])
    bbox = (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
    net = RoadNetwork(trip.nodes, trip.arcs, proj, bbox)
    cands = net.candidates(xy)
    return TripContext(trip, track, rep, xy, net, cands)


def gps_heading_speed(xy: np.ndarray, t: np.ndarray):
    """Central-difference heading and speed for every observation."""
    n = len(xy)
    i0 = np.clip(np.arange(n) - 1, 0, n - 1)
    i1 = np.clip(np.arange(n) + 1, 0, n - 1)
    d = xy[i1] - xy[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    sp = np.hypot(d[:, 0], d[:, 1]) / dt
    return bearing(d[:, 0], d[:, 1]), sp


def build_features(ctx: TripContext) -> pd.DataFrame:
    xy, t, net = ctx.xy, ctx.track[:, 2], ctx.net
    n = len(xy)
    head, speed = gps_heading_speed(xy, t)
    dt = np.diff(t)
    dt_prev = np.r_[0.0, dt]
    dt_next = np.r_[dt, 0.0]

    sizes = np.array([len(c[0]) for c in ctx.cands])
    pidx = np.repeat(np.arange(n), sizes)
    arcs = np.concatenate([c[0] for c in ctx.cands]).astype(np.int64)
    dist = np.concatenate([c[1] for c in ctx.cands])
    tt = np.concatenate([c[2] for c in ctx.cands])
    rank = np.concatenate([np.arange(k) for k in sizes]).astype(float)
    dmin = np.repeat([c[1].min() if len(c[1]) else 0 for c in ctx.cands], sizes)
    g = np.exp(-0.5 * (dist / GAUSS_SIGMA) ** 2)
    gsum = np.bincount(pidx, weights=g, minlength=n)[pidx]

    A, B = net.A[arcs], net.B[arcs]
    # window distances of neighbouring observations to the same arc
    win = []
    for off in range(-WIN, WIN + 1):
        if off == 0:
            continue
        j = np.clip(pidx + off, 0, n - 1)
        d, _ = point_segment(xy[j], A, B)
        win.append(d)
    win = np.vstack(win)
    dist_prev = win[WIN - 1]
    dist_next = win[WIN]
    allw = np.vstack([win, dist[None]])
    # rank of this arc's distance within its group, averaged across window
    # (cheap proxy: distance relative to group minimum at each offset)
    win_rank = np.mean(allw - dmin[None], axis=0)

    bh = net.bearing[arcs]
    hd = angle_diff(head[pidx], bh)
    hdr = angle_diff(head[pidx], bh + np.pi)

    df = pd.DataFrame({
        "trip": ctx.trip.trip_id,
        "point": pidx.astype(np.int32),
        "arc": arcs,
        "dist": dist,
        "log_dist": np.log1p(dist),
        "rank": rank,
        "dist_minus_min": dist - dmin,
        "dist_ratio_min": (dist + 1.0) / (dmin + 1.0),
        "gauss_share": g / np.maximum(gsum, 1e-12),
        "n_cand": sizes[pidx].astype(float),
        "t": tt,
        "at_end": ((tt <= 0) | (tt >= 1)).astype(float),
        "arc_len": net.length[arcs],
        "log_arc_len": np.log1p(net.length[arcs]),
        "head_diff": hd,
        "cos_head": np.cos(hd),
        "head_diff_rev": hdr,
        "speed": speed[pidx],
        "heading_reliable": (speed[pidx] > 2.0).astype(float),
        "two_way": net.has_reverse[arcs].astype(float),
        "dist_prev": dist_prev,
        "dist_next": dist_next,
        "win_mean_dist": allw.mean(0),
        "win_max_dist": allw.max(0),
        "win_rank_mean": win_rank,
        "out_deg_to": net.out_deg[net.arcs[arcs, 1]].astype(float),
        "in_deg_from": net.in_deg[net.arcs[arcs, 0]].astype(float),
        "dt_prev": dt_prev[pidx],
        "dt_next": dt_next[pidx],
    })
    for c in FEATURES:
        df[c] = df[c].astype(np.float32)
    return df


def attach_labels(df: pd.DataFrame, ctx: TripContext, label_arc: np.ndarray) -> pd.DataFrame:
    canon = ctx.net.canon
    df["label"] = (canon[df["arc"].to_numpy()] == canon[label_arc[df["point"].to_numpy()]]).astype(np.int8)
    return df
