"""Raw-file readers.

File formats (verified in scripts/01_eda.py):
  <id>.track  lon lat t          (whitespace separated, no header)
  <id>.nodes  lon lat            (0-based row index = node id)
  <id>.arcs   from_node to_node  (0-based row index = arc id)
  <id>.route  arc_id             (chronological ground-truth arc sequence)

The prediction pipeline only ever calls `load_trip(..., with_route=False)`; the
.route file is read exclusively by evaluation code (`load_route`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA_DIR


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray   # (N, 3) lon, lat, t
    nodes: np.ndarray   # (V, 2) lon, lat
    arcs: np.ndarray    # (E, 2) from, to


def trip_dir(partition: str, trip_id: str) -> Path:
    return DATA_DIR / partition / trip_id


def list_trips(partition: str) -> list[str]:
    d = DATA_DIR / partition
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def _load(path: Path, dtype=float, ndmin=2) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 2 if ndmin == 2 else 0), dtype=dtype)
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_trip(partition: str, trip_id: str) -> Trip:
    d = trip_dir(partition, trip_id)
    track = _load(d / f"{trip_id}.track")
    nodes = _load(d / f"{trip_id}.nodes")
    arcs = _load(d / f"{trip_id}.arcs", dtype=np.int64)
    return Trip(trip_id, partition, track, nodes, arcs)


def load_route(partition: str, trip_id: str) -> np.ndarray:
    """Ground-truth arc sequence. Evaluation use only."""
    return _load(trip_dir(partition, trip_id) / f"{trip_id}.route", dtype=np.int64, ndmin=1)


def clean_track(track: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remove invalid / duplicate observations and sort by time.

    Returns the cleaned track and a small report of what was removed.
    """
    rep = {"n_raw": int(len(track))}
    ok = np.all(np.isfinite(track), axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    ok &= ~((track[:, 0] == 0) & (track[:, 1] == 0))
    rep["n_invalid"] = int((~ok).sum())
    track = track[ok]
    order = np.argsort(track[:, 2], kind="stable")
    rep["n_unsorted"] = int((order != np.arange(len(order))).sum())
    track = track[order]
    # drop duplicated timestamps (keep first)
    if len(track):
        keep = np.r_[True, np.diff(track[:, 2]) > 0]
        rep["n_dup_time"] = int((~keep).sum())
        track = track[keep]
    else:
        rep["n_dup_time"] = 0
    rep["n_clean"] = int(len(track))
    return track, rep
