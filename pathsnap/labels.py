"""Derive a per-observation reference arc from the trip-level .route file.

The .route file is an ordered list of arcs, not one label per GPS row. We align
the two sequences with a monotone dynamic programme: each GPS point i gets a
route position j(i) with j non-decreasing in i, minimising the total
point-to-arc distance. Route arcs may be skipped between consecutive GPS
points (short arcs passed within one second), but the order is never reversed,
which resolves out-and-back and loop sections that a plain nearest-arc
assignment would confuse.

Used only for training labels and server-side evaluation.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment


def align_route(pts_xy: np.ndarray, route: np.ndarray, net) -> tuple[np.ndarray, np.ndarray]:
    """Return (label_arc per point, distance to that arc)."""
    n, m = len(pts_xy), len(route)
    if n == 0 or m == 0:
        return np.full(n, -1), np.full(n, np.inf)
    A, B = net.A[route], net.B[route]
    D = np.empty((n, m))
    for s in range(0, n, 512):
        p = pts_xy[s:s + 512]
        pp = np.repeat(p, m, 0)
        d, _ = point_segment(pp, np.tile(A, (len(p), 1)), np.tile(B, (len(p), 1)))
        D[s:s + 512] = d.reshape(len(p), m)
    cost = D[0].copy()
    back = np.empty((n, m), dtype=np.int32)
    back[0] = np.arange(m)
    idx = np.arange(m)
    for i in range(1, n):
        # best predecessor j' <= j  == running minimum of the previous row
        run_min = np.minimum.accumulate(cost)
        back[i] = np.maximum.accumulate(np.where(cost <= run_min, idx, 0))
        cost = run_min + D[i]
    j = int(np.argmin(cost))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    return route[path], D[np.arange(n), path]
