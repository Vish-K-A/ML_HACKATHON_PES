"""
Exploratory Data Analysis (EDA) and Data Quality Inspection for PathSnap.
Generates trip inventories, partition summaries, and road-network topology statistics.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from data_processing.preprocess import load_trip


def run_eda(student_folder: str, output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for partition in ["train", "student_test"]:
        part_dir = os.path.join(student_folder, partition)
        if not os.path.isdir(part_dir):
            continue
        for trip_id in sorted(os.listdir(part_dir)):
            tdir = os.path.join(part_dir, trip_id)
            if not os.path.isdir(tdir):
                continue

            trip = load_trip(tdir, trip_id=trip_id, partition=partition, load_route=True)
            track = trip.track
            nodes = trip.nodes
            arcs = trip.arcs
            route = trip.route

            dt = np.diff(track[:, 2]) if len(track) > 1 else np.array([1.0])
            step_dists = np.linalg.norm(np.diff(trip.track_xy, axis=0), axis=1) if len(track) > 1 else np.array([0.0])
            traj_len_m = float(np.sum(step_dists))

            # Road-network connectivity & isolated nodes
            n_nodes = len(nodes)
            n_arcs = len(arcs)
            adj = csr_matrix((np.ones(n_arcs, dtype=np.int8), (arcs[:, 0], arcs[:, 1])), shape=(n_nodes, n_nodes))
            n_wcc, wcc_labels = connected_components(adj, directed=True, connection="weak")
            n_scc, _ = connected_components(adj, directed=True, connection="strong")
            deg_total = np.bincount(arcs[:, 0], minlength=n_nodes) + np.bincount(arcs[:, 1], minlength=n_nodes)
            isolated_nodes = int(np.sum(deg_total == 0))

            # Bidirectional arc check
            arc_hashes = set(arcs[:, 0].astype(np.int64) * (n_nodes + 1) + arcs[:, 1].astype(np.int64))
            rev_hashes = arcs[:, 1].astype(np.int64) * (n_nodes + 1) + arcs[:, 0].astype(np.int64)
            bidir_count = int(sum(1 for h in rev_hashes if h in arc_hashes))

            # Route continuity in ground truth
            n_route = len(route) if route is not None else 0
            route_conn = 0
            if route is not None and len(route) > 1:
                route_conn = int(np.sum(arcs[route[:-1], 1] == arcs[route[1:], 0]))

            records.append({
                "partition": partition,
                "trip_id": trip_id,
                "n_gps": len(track),
                "n_nodes": n_nodes,
                "n_arcs": n_arcs,
                "n_route_segments": n_route,
                "traj_length_km": round(traj_len_m / 1000.0, 3),
                "duration_s": float(track[-1, 2] - track[0, 2]),
                "dt_1s_fraction": round(float(np.mean(np.isclose(dt, 1.0))), 4),
                "dt_min_s": float(np.min(dt)),
                "dt_max_s": float(np.max(dt)),
                "n_gaps_gt_1s": int(np.sum(dt > 1.001)),
                "n_gaps_gt_5s": int(np.sum(dt > 5.0)),
                "n_gaps_gt_10s": int(np.sum(dt > 10.0)),
                "ref_lon": round(trip.ref_lon, 5),
                "ref_lat": round(trip.ref_lat, 5),
                "lon_scale_m_per_deg": round(trip.lon_scale, 2),
                "lat_scale_m_per_deg": round(trip.lat_scale, 2),
                "n_weak_components": int(n_wcc),
                "n_strong_components": int(n_scc),
                "isolated_nodes": isolated_nodes,
                "bidirectional_arc_ratio": round(bidir_count / max(1, n_arcs), 4),
                "gt_route_continuity": round(route_conn / max(1, n_route - 1), 4),
                "missing_values": int(np.isnan(track).sum() + np.isnan(nodes).sum()),
            })

    df = pd.DataFrame(records)
    inv_path = os.path.join(output_dir, "dataset_inventory.csv")
    df.to_csv(inv_path, index=False)

    # Partition summary table
    summary_cols = [
        "n_gps", "n_nodes", "n_arcs", "n_route_segments",
        "traj_length_km", "duration_s", "dt_1s_fraction", "dt_max_s",
        "n_gaps_gt_1s", "n_gaps_gt_5s", "isolated_nodes",
        "bidirectional_arc_ratio", "gt_route_continuity",
    ]
    part_summary = df.groupby("partition")[summary_cols].agg(["count", "mean", "std", "min", "median", "max"])
    part_summary.to_csv(os.path.join(output_dir, "partition_summary.csv"))

    quality_report = {
        "total_trips_inspected": int(len(df)),
        "train_trips": int((df["partition"] == "train").sum()),
        "student_test_trips": int((df["partition"] == "student_test").sum()),
        "total_train_gps_observations": int(df.loc[df["partition"] == "train", "n_gps"].sum()),
        "total_student_test_gps_observations": int(df.loc[df["partition"] == "student_test", "n_gps"].sum()),
        "total_missing_or_nan_values": int(df["missing_values"].sum()),
        "mean_1hz_sampling_rate_train": round(float(df.loc[df["partition"] == "train", "dt_1s_fraction"].mean()), 4),
        "mean_1hz_sampling_rate_student_test": round(float(df.loc[df["partition"] == "student_test", "dt_1s_fraction"].mean()), 4),
        "trips_with_gaps_gt_5s": int((df["n_gaps_gt_5s"] > 0).sum()),
        "max_time_gap_seconds": float(df["dt_max_s"].max()),
        "trips_with_isolated_nodes": int((df["isolated_nodes"] > 0).sum()),
        "mean_bidirectional_arc_ratio": round(float(df["bidirectional_arc_ratio"].mean()), 4),
        "mean_gt_route_continuity": round(float(df["gt_route_continuity"].mean()), 4),
        "global_lon_range": [round(float(df["ref_lon"].min()), 4), round(float(df["ref_lon"].max()), 4)],
        "global_lat_range": [round(float(df["ref_lat"].min()), 4), round(float(df["ref_lat"].max()), 4)],
    }
    with open(os.path.join(output_dir, "data_quality_summary.json"), "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    return df


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    student_dir = os.path.join(root, "Student folder")
    out_dir = os.path.join(root, "supporting_results")
    df_eda = run_eda(student_dir, out_dir)
    print(f"EDA completed on {len(df_eda)} trips. Saved to {out_dir}")
