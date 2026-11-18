"""
Candidate Generation and Spatio-Temporal / Topological Feature Engineering for PathSnap.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

from data_processing.preprocess import TripData, align_reference_route, compute_kinematics


FEATURE_NAMES: List[str] = [
    "perp_dist_m",
    "dist_to_u_m",
    "dist_to_v_m",
    "proj_fraction",
    "is_interior_proj",
    "seg_length_m",
    "dist_diff_min_m",
    "dist_ratio_min",
    "dist_zscore",
    "dist_rank_norm",
    "cos_heading_inst",
    "cos_heading_8m",
    "cos_heading_25m",
    "cos_heading_50m",
    "cos_heading_blend",
    "sin_heading_blend",
    "heading_diff_best",
    "combined_geom_score",
    "combined_score_diff_min",
    "perp_dist_prev3_m",
    "perp_dist_next3_m",
    "proj_advance_3s",
    "prev_step_connected",
    "next_step_connected",
    "node_v_out_degree",
    "gps_speed_ms",
    "gps_dt_s",
]


def point_to_segments_geometry(
    pts_xy: np.ndarray,
    u_seg: np.ndarray,
    ab_seg: np.ndarray,
    len2_seg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized point-to-segment distance and normalized projection parameter tau in [0, 1].
    pts_xy: (T, 2)
    u_seg: (T, M, 2)
    ab_seg: (T, M, 2)
    len2_seg: (T, M)
    Returns:
        dist: (T, M) perpendicular distance in meters
        tau: (T, M) normalized projection coordinate along u -> v in [0, 1]
    """
    ap = pts_xy[:, None, :] - u_seg
    tau = np.sum(ap * ab_seg, axis=2) / np.maximum(len2_seg, 1e-12)
    tau_clipped = np.clip(tau, 0.0, 1.0)
    proj = u_seg + tau_clipped[:, :, None] * ab_seg
    dist = np.linalg.norm(pts_xy[:, None, :] - proj, axis=2)
    return dist, tau_clipped


def extract_trip_candidates_and_features(
    trip: TripData,
    k_candidates: int = 8,
    presearch_m: int = 48,
    include_labels: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Generates top-K candidate road segments for each GPS observation in `trip`
    and extracts 27 spatial, kinematic, contextual, and topological features per candidate.
    """
    T = len(trip.track)
    A = len(trip.arcs)
    k_candidates = min(k_candidates, A)
    presearch_m = min(presearch_m, A)

    kin = compute_kinematics(trip.track, trip.track_xy)

    # Precompute all road segment vectors in local meters
    u_all = trip.nodes_xy[trip.arcs[:, 0]]
    v_all = trip.nodes_xy[trip.arcs[:, 1]]
    ab_all = v_all - u_all
    seg_len_all = np.maximum(np.linalg.norm(ab_all, axis=1), 1e-6)
    ab_unit_all = ab_all / seg_len_all[:, None]
    len2_all = seg_len_all ** 2
    mid_all = 0.5 * (u_all + v_all)

    # Out-degree of each node (excluding immediate reverse edge u->v->u)
    n_nodes = len(trip.nodes)
    out_deg = np.bincount(trip.arcs[:, 0], minlength=n_nodes).astype(np.float64)

    # Spatial index on segment midpoints for fast pre-filtering
    tree = cKDTree(mid_all)
    _, pre_idxs = tree.query(trip.track_xy, k=presearch_m)
    if pre_idxs.ndim == 1:
        pre_idxs = pre_idxs[:, None]

    u_pre = u_all[pre_idxs]
    ab_pre = ab_all[pre_idxs]
    len2_pre = len2_all[pre_idxs]

    d_pre, _ = point_to_segments_geometry(trip.track_xy, u_pre, ab_pre, len2_pre)
    cos_pre = np.sum(kin["h_blend"][:, None, :] * ab_unit_all[pre_idxs], axis=2)

    # Rank pre-candidates by distance + slight heading tie-breaker so both directions of close streets are kept
    rank_score = d_pre + 1.5 * (1.0 - cos_pre)
    order = np.argsort(rank_score, axis=1)[:, :k_candidates]
    cand_arcs = np.take_along_axis(pre_idxs, order, axis=1)  # (T, K)

    # Gather segment geometry for selected (T, K) candidates
    u_c = u_all[cand_arcs]             # (T, K, 2)
    v_c = v_all[cand_arcs]             # (T, K, 2)
    ab_c = ab_all[cand_arcs]           # (T, K, 2)
    ab_unit_c = ab_unit_all[cand_arcs] # (T, K, 2)
    seg_len_c = seg_len_all[cand_arcs] # (T, K)
    len2_c = len2_all[cand_arcs]       # (T, K)

    perp_dist, proj_frac = point_to_segments_geometry(trip.track_xy, u_c, ab_c, len2_c)
    dist_to_u = np.linalg.norm(trip.track_xy[:, None, :] - u_c, axis=2)
    dist_to_v = np.linalg.norm(trip.track_xy[:, None, :] - v_c, axis=2)
    is_interior = ((proj_frac > 0.02) & (proj_frac < 0.98)).astype(np.float64)

    # Competition features within timestamp t
    min_dist = np.min(perp_dist, axis=1, keepdims=True)
    dist_diff_min = perp_dist - min_dist
    dist_ratio_min = perp_dist / (min_dist + 1.0)
    mean_dist = np.mean(perp_dist, axis=1, keepdims=True)
    std_dist = np.std(perp_dist, axis=1, keepdims=True)
    dist_zscore = (perp_dist - mean_dist) / (std_dist + 0.5)

    dist_order = np.argsort(np.argsort(perp_dist, axis=1), axis=1)
    dist_rank_norm = dist_order / max(1, k_candidates - 1)

    # Multi-scale heading cosine similarities
    cos_inst = np.sum(kin["h_inst"][:, None, :] * ab_unit_c, axis=2)
    cos_8m = np.sum(kin["h_8m"][:, None, :] * ab_unit_c, axis=2)
    cos_25m = np.sum(kin["h_25m"][:, None, :] * ab_unit_c, axis=2)
    cos_50m = np.sum(kin["h_50m"][:, None, :] * ab_unit_c, axis=2)
    cos_blend = np.sum(kin["h_blend"][:, None, :] * ab_unit_c, axis=2)
    sin_blend = np.sqrt(np.maximum(0.0, 1.0 - np.clip(cos_blend, -1.0, 1.0) ** 2))
    heading_diff_best = np.max(cos_blend, axis=1, keepdims=True) - cos_blend

    combined_geom = perp_dist + 6.0 * (1.0 - cos_blend)
    combined_diff_min = combined_geom - np.min(combined_geom, axis=1, keepdims=True)

    # Look-back (t-3) and look-ahead (t+3) projection & perpendicular distances
    idx_prev3 = np.maximum(0, np.arange(T) - 3)
    idx_next3 = np.minimum(T - 1, np.arange(T) + 3)
    perp_prev3, tau_prev3 = point_to_segments_geometry(trip.track_xy[idx_prev3], u_c, ab_c, len2_c)
    perp_next3, tau_next3 = point_to_segments_geometry(trip.track_xy[idx_next3], u_c, ab_c, len2_c)
    proj_advance_3s = tau_next3 - tau_prev3

    # Local topological connectivity to adjacent step top-3 candidates
    top3_mask = combined_diff_min <= np.sort(combined_diff_min, axis=1)[:, min(2, k_candidates - 1) : min(3, k_candidates)]
    cand_u_node = trip.arcs[cand_arcs, 0]  # (T, K)
    cand_v_node = trip.arcs[cand_arcs, 1]  # (T, K)

    prev_conn = np.zeros((T, k_candidates), dtype=np.float64)
    next_conn = np.zeros((T, k_candidates), dtype=np.float64)
    prev_conn[0] = 1.0
    next_conn[-1] = 1.0

    if T > 1:
        # Vectorized check between (t-1, k_prev) and (t, k_curr)
        pa = cand_arcs[:-1, :, None]      # (T-1, K_prev, 1)
        ca = cand_arcs[1:, None, :]       # (T-1, 1, K_curr)
        pu = cand_u_node[:-1, :, None]
        pv = cand_v_node[:-1, :, None]
        cu = cand_u_node[1:, None, :]
        cv = cand_v_node[1:, None, :]

        same_seg = (pa == ca)
        forward_adj = (pv == cu) & (cv != pu)
        valid_link = same_seg | forward_adj  # (T-1, K_prev, K_curr)

        prev_top_valid = valid_link & top3_mask[:-1, :, None]
        next_top_valid = valid_link & top3_mask[1:, None, :]

        prev_conn[1:] = np.any(prev_top_valid, axis=1).astype(np.float64)
        next_conn[:-1] = np.any(next_top_valid, axis=2).astype(np.float64)

    v_out_deg = out_deg[cand_v_node]
    speed_feat = np.broadcast_to(kin["speed_ms"][:, None], (T, k_candidates))
    dt_feat = np.broadcast_to(kin["dt_prev"][:, None], (T, k_candidates))

    X = np.stack(
        [
            perp_dist,
            dist_to_u,
            dist_to_v,
            proj_frac,
            is_interior,
            seg_len_c,
            dist_diff_min,
            dist_ratio_min,
            dist_zscore,
            dist_rank_norm,
            cos_inst,
            cos_8m,
            cos_25m,
            cos_50m,
            cos_blend,
            sin_blend,
            heading_diff_best,
            combined_geom,
            combined_diff_min,
            perp_prev3,
            perp_next3,
            proj_advance_3s,
            prev_conn,
            next_conn,
            v_out_deg,
            speed_feat,
            dt_feat,
        ],
        axis=-1,
    ).astype(np.float32)  # (T, K, F)

    result = {
        "X": X,                       # (T, K, F)
        "cand_arcs": cand_arcs,       # (T, K)
        "perp_dist": perp_dist,       # (T, K)
        "proj_frac": proj_frac,       # (T, K)
        "cos_blend": cos_blend,       # (T, K)
        "dt_prev": kin["dt_prev"],    # (T,)
        "speed_ms": kin["speed_ms"],  # (T,)
    }

    if include_labels and trip.route is not None:
        gt_arcs = align_reference_route(trip, kinematics=kin)  # (T,)
        y = (cand_arcs == gt_arcs[:, None]).astype(np.int32)   # (T, K)
        result["gt_arcs"] = gt_arcs
        result["y"] = y

    return result
