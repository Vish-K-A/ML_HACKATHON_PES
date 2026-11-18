"""
Data Loading, Coordinate Projection, Kinematic Extraction, and Reference Route Alignment
for PathSnap Distributed Map-Matching & Route Reconstruction.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


# WGS-84 Ellipsoid Constants
WGS84_A = 6378137.0            # Semi-major axis (meters)
WGS84_E2 = 0.00669437999014    # First eccentricity squared


@dataclass
class TripData:
    trip_id: str
    partition: str
    track: np.ndarray          # (T, 3): [lon, lat, timestamp]
    nodes: np.ndarray          # (N, 2): [lon, lat]
    arcs: np.ndarray           # (A, 2): [from_node_id, to_node_id]
    route: Optional[np.ndarray]  # (M,) ground-truth arc_ids if available
    nodes_xy: np.ndarray       # (N, 2) projected meters [x_east, y_north]
    track_xy: np.ndarray       # (T, 2) projected meters [x_east, y_north]
    ref_lon: float
    ref_lat: float
    lon_scale: float
    lat_scale: float


def compute_wgs84_scales(lat_deg: float) -> Tuple[float, float]:
    """
    Computes exact WGS-84 meridional (north-south) and prime-vertical (east-west)
    meters-per-degree scaling factors at reference latitude `lat_deg`.
    """
    phi = np.radians(lat_deg)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    denom = np.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)

    # Prime vertical radius of curvature N(phi) * cos(phi) -> meters per radian of longitude
    n_rad = WGS84_A / denom
    lon_scale = (np.pi / 180.0) * n_rad * cos_phi

    # Meridional radius of curvature M(phi) -> meters per radian of latitude
    m_rad = WGS84_A * (1.0 - WGS84_E2) / (denom ** 3)
    lat_scale = (np.pi / 180.0) * m_rad

    return float(lon_scale), float(lat_scale)


def load_trip(trip_dir: str, trip_id: Optional[str] = None, partition: str = "unknown", load_route: bool = True) -> TripData:
    """
    Loads a single trip's .track, .nodes, .arcs, and optionally .route files,
    and projects geographical coordinates (lon, lat) into local tangent-plane meters (x, y).
    """
    if trip_id is None:
        trip_id = os.path.basename(os.path.normpath(trip_dir))

    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    track = np.loadtxt(track_path, dtype=np.float64)
    if track.ndim == 1:
        track = track.reshape(1, -1)

    nodes = np.loadtxt(nodes_path, dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)

    arcs = np.loadtxt(arcs_path, dtype=np.int64)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)

    # Ensure chronological timestamp ordering in track
    if len(track) > 1 and np.any(np.diff(track[:, 2]) < 0):
        order = np.argsort(track[:, 2], kind="mergesort")
        track = track[order]

    route = None
    if load_route and os.path.exists(route_path):
        route = np.loadtxt(route_path, dtype=np.int64)
        if route.ndim == 0:
            route = route.reshape(1)

    ref_lon = float(np.mean(track[:, 0]))
    ref_lat = float(np.mean(track[:, 1]))
    lon_scale, lat_scale = compute_wgs84_scales(ref_lat)

    nodes_xy = np.column_stack([
        (nodes[:, 0] - ref_lon) * lon_scale,
        (nodes[:, 1] - ref_lat) * lat_scale,
    ])
    track_xy = np.column_stack([
        (track[:, 0] - ref_lon) * lon_scale,
        (track[:, 1] - ref_lat) * lat_scale,
    ])

    return TripData(
        trip_id=trip_id,
        partition=partition,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
        nodes_xy=nodes_xy,
        track_xy=track_xy,
        ref_lon=ref_lon,
        ref_lat=ref_lat,
        lon_scale=lon_scale,
        lat_scale=lat_scale,
    )


def compute_displacement_heading(track_xy: np.ndarray, disp_m: float) -> np.ndarray:
    """
    Vectorized computation of unit heading vectors at each GPS point by finding
    backward and forward trajectory points separated by at least `disp_m` meters.
    Robust to stationary idling, traffic stops, and high-frequency GPS jitter.
    """
    T = len(track_xy)
    vel = np.zeros((T, 2), dtype=np.float64)
    if T <= 1:
        return vel

    step_d = np.linalg.norm(np.diff(track_xy, axis=0), axis=1)
    cum_s = np.concatenate([[0.0], np.cumsum(step_d)])

    # Use cumulative path distance (with 1.25x factor for jitter) and refine vectorially
    b_idx = np.searchsorted(cum_s, cum_s - disp_m * 1.25, side="right") - 1
    b_idx = np.clip(b_idx, 0, T - 1)
    f_idx = np.searchsorted(cum_s, cum_s + disp_m * 1.25, side="left")
    f_idx = np.clip(f_idx, 0, T - 1)

    # Refine any points where Euclidean displacement is still < disp_m (e.g. parked jitter)
    for step_mult in (2.5, 5.0, 10.0, 25.0):
        need_b = (b_idx > 0) & (np.linalg.norm(track_xy - track_xy[b_idx], axis=1) < disp_m * 0.7)
        if np.any(need_b):
            b_idx[need_b] = np.clip(
                np.searchsorted(cum_s, cum_s[need_b] - disp_m * step_mult, side="right") - 1, 0, T - 1
            )
        need_f = (f_idx < T - 1) & (np.linalg.norm(track_xy[f_idx] - track_xy, axis=1) < disp_m * 0.7)
        if np.any(need_f):
            f_idx[need_f] = np.clip(
                np.searchsorted(cum_s, cum_s[need_f] + disp_m * step_mult, side="left"), 0, T - 1
            )

    v = track_xy[f_idx] - track_xy[b_idx]
    norms = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
    return v / norms


def compute_kinematics(track: np.ndarray, track_xy: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extracts time gaps, instantaneous speed, smoothed velocity, and multi-scale
    trajectory heading vectors (8m, 25m, 50m, and blended).
    """
    T = len(track)
    timestamps = track[:, 2]

    dt_prev = np.ones(T, dtype=np.float64)
    dt_next = np.ones(T, dtype=np.float64)
    if T > 1:
        diffs = np.maximum(np.diff(timestamps), 1e-3)
        dt_prev[1:] = diffs
        dt_prev[0] = diffs[0]
        dt_next[:-1] = diffs
        dt_next[-1] = diffs[-1]

    disp_prev = np.zeros(T, dtype=np.float64)
    disp_next = np.zeros(T, dtype=np.float64)
    if T > 1:
        step_dist = np.linalg.norm(np.diff(track_xy, axis=0), axis=1)
        disp_prev[1:] = step_dist
        disp_prev[0] = step_dist[0]
        disp_next[:-1] = step_dist
        disp_next[-1] = step_dist[-1]

    speed_ms = 0.5 * (disp_prev / dt_prev + disp_next / dt_next)

    # Instantaneous central-difference heading (+/- 2 steps) - vectorized
    idx0 = np.maximum(0, np.arange(T) - 2)
    idx1 = np.minimum(T - 1, np.arange(T) + 2)
    v_inst = track_xy[idx1] - track_xy[idx0]
    vel_inst = v_inst / np.maximum(np.linalg.norm(v_inst, axis=1, keepdims=True), 1e-6)

    h_8m = compute_displacement_heading(track_xy, 8.0)
    h_25m = compute_displacement_heading(track_xy, 25.0)
    h_50m = compute_displacement_heading(track_xy, 50.0)

    h_blend = 0.35 * h_8m + 0.40 * h_25m + 0.25 * h_50m
    h_blend_norm = np.maximum(np.linalg.norm(h_blend, axis=1, keepdims=True), 1e-6)
    h_blend = h_blend / h_blend_norm

    return {
        "dt_prev": dt_prev,
        "dt_next": dt_next,
        "speed_ms": speed_ms,
        "h_inst": vel_inst,
        "h_8m": h_8m,
        "h_25m": h_25m,
        "h_50m": h_50m,
        "h_blend": h_blend,
    }


def align_reference_route(trip: TripData, kinematics: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """
    Associates each GPS observation in `trip.track` with its corresponding reference road segment
    (`arc_id`) from `trip.route` using multi-scale heading-aware monotonic alignment along `trip.route`.
    NOTE: Used strictly for training label generation and development/server evaluation.
    """
    if trip.route is None or len(trip.route) == 0:
        raise ValueError(f"Trip {trip.trip_id} does not have reference route data.")

    if kinematics is None:
        kinematics = compute_kinematics(trip.track, trip.track_xy)

    route = trip.route
    u_r = trip.nodes_xy[trip.arcs[route, 0]]
    v_r = trip.nodes_xy[trip.arcs[route, 1]]
    ab_r = v_r - u_r
    ab_len = np.maximum(np.linalg.norm(ab_r, axis=1, keepdims=True), 1e-6)
    ab_unit = ab_r / ab_len
    ab_len2 = np.sum(ab_r ** 2, axis=1)

    ap = trip.track_xy[:, None, :] - u_r[None, :, :]
    tau = np.clip(np.sum(ap * ab_r[None, :, :], axis=2) / np.maximum(ab_len2[None, :], 1e-12), 0.0, 1.0)
    proj = u_r[None, :, :] + tau[:, :, None] * ab_r[None, :, :]
    d_route = np.linalg.norm(trip.track_xy[:, None, :] - proj, axis=2)  # (T, M)

    h_blend = kinematics["h_blend"]
    cos_sim = h_blend @ ab_unit.T  # (T, M)

    # Combine perpendicular distance (m) and directional alignment cost
    step_cost = d_route + 6.0 * (1.0 - cos_sim)
    T, M = step_cost.shape

    dp = np.full((T, M), np.inf, dtype=np.float64)
    back = np.zeros((T, M), dtype=np.int32)
    init_span = min(M, 25)
    dp[0, :init_span] = step_cost[0, :init_span]

    max_jump = min(M, 15)
    for i in range(1, T):
        best_prev = dp[i - 1].copy()
        best_k = np.zeros(M, dtype=np.int32)
        for k in range(1, max_jump):
            cand = np.empty(M, dtype=np.float64)
            cand[:k] = np.inf
            skip_pen = 0.2 * k if k <= 2 else 1.5 * (k - 1)
            cand[k:] = dp[i - 1, :-k] + skip_pen
            mask = cand < best_prev
            best_prev[mask] = cand[mask]
            best_k[mask] = k
        dp[i] = step_cost[i] + best_prev
        back[i] = np.arange(M, dtype=np.int32) - best_k

    mono_pos = np.zeros(T, dtype=np.int32)
    mono_pos[-1] = int(np.argmin(dp[-1]))
    for i in range(T - 1, 0, -1):
        mono_pos[i - 1] = back[i, mono_pos[i]]

    return route[mono_pos]
