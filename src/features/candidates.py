"""Bounded road candidate generation and route-label alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data.pathsnap import Trip, project_trip

FEATURES = ["distance_m", "heading_difference_deg", "speed_mps", "time_gap_s"]


def _point_segment_distance(point: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    vectors = ends - starts
    denom = np.einsum("ij,ij->i", vectors, vectors)
    denom = np.maximum(denom, 1e-9)
    t = np.einsum("ij,ij->i", point - starts, vectors) / denom
    t = np.clip(t, 0.0, 1.0)
    nearest = starts + t[:, None] * vectors
    return np.linalg.norm(nearest - point, axis=1)


def _bearing(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    delta = ends - starts
    return np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) % 360.0


def _align_route(track_xy: np.ndarray, route_starts: np.ndarray, route_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate route arc per GPS point; enforce nondecreasing route order."""
    labels = np.empty(track_xy.shape[0], dtype=np.int64)
    errors = np.empty(track_xy.shape[0], dtype=np.float64)
    floor_idx = 0
    for i, point in enumerate(track_xy):
        distances = _point_segment_distance(point, route_starts[floor_idx:], route_ends[floor_idx:])
        local = int(np.argmin(distances))
        floor_idx += local
        labels[i] = floor_idx
        errors[i] = distances[local]
    return labels, errors


def trip_candidates(
    trip: Trip,
    max_candidates: int = 8,
    radius_m: float = 100.0,
    midpoint_prefilter: int = 40,
    max_negatives: int = 7,
) -> tuple[pd.DataFrame, dict[str, float]]:
    nodes_xy, track_xy = project_trip(trip)
    starts = nodes_xy[trip.arcs[:, 0]]
    ends = nodes_xy[trip.arcs[:, 1]]
    midpoints = (starts + ends) / 2.0
    tree = cKDTree(midpoints)
    route_arc_ids = trip.route
    aligned_route_positions, alignment_errors = _align_route(
        track_xy, starts[route_arc_ids], ends[route_arc_ids]
    )
    true_arc_ids = route_arc_ids[aligned_route_positions]
    dt = np.diff(trip.track[:, 2], prepend=trip.track[0, 2])
    displacement = np.linalg.norm(np.diff(track_xy, axis=0, prepend=track_xy[[0]]), axis=1)
    speed = np.divide(displacement, dt, out=np.zeros_like(displacement), where=dt > 0)
    movement_heading = np.degrees(np.arctan2(
        np.diff(track_xy[:, 0], prepend=track_xy[0, 0]),
        np.diff(track_xy[:, 1], prepend=track_xy[0, 1]),
    )) % 360.0

    rows: list[dict[str, float | int | str]] = []
    candidate_hits = 0
    for obs_idx, point in enumerate(track_xy):
        k = min(midpoint_prefilter, len(midpoints))
        _, nearby = tree.query(point, k=k)
        nearby = np.atleast_1d(nearby).astype(np.int64)
        distances = _point_segment_distance(point, starts[nearby], ends[nearby])
        order = np.argsort(distances)
        order = order[distances[order] <= radius_m][:max_candidates]
        if order.size == 0:
            order = np.array([int(np.argmin(distances))])
        arc_ids = nearby[order]
        dists = distances[order]
        headings = _bearing(starts[arc_ids], ends[arc_ids])
        hdiff = np.abs((headings - movement_heading[obs_idx] + 180.0) % 360.0 - 180.0)
        positive_pos = np.flatnonzero(arc_ids == true_arc_ids[obs_idx])
        if positive_pos.size:
            candidate_hits += 1
            keep = np.unique(np.r_[positive_pos[0], np.argsort(dists)[:max_negatives]])
        else:
            keep = np.arange(len(arc_ids))
        for j in keep:
            rows.append({
                "trip_id": trip.trip_id,
                "observation": obs_idx,
                "arc_id": int(arc_ids[j]),
                "reference_arc_id": int(true_arc_ids[obs_idx]),
                "distance_m": float(dists[j]),
                "heading_difference_deg": float(hdiff[j]),
                "speed_mps": float(speed[obs_idx]),
                "time_gap_s": float(max(dt[obs_idx], 0.0)),
                "label": int(arc_ids[j] == true_arc_ids[obs_idx]),
            })
    frame = pd.DataFrame(rows)
    summary = {
        "observations": float(len(track_xy)),
        "candidate_recall": candidate_hits / max(len(track_xy), 1),
        "alignment_median_m": float(np.median(alignment_errors)),
        "alignment_p95_m": float(np.percentile(alignment_errors, 95)),
    }
    return frame, summary
