"""
Generates all EDA charts, Model Comparison plots, Confusion Matrices, Epoch-wise MLP
Training Curves, Bayesian Optimisation Convergence plots, HMM+Viterbi comparisons,
and Interactive Folium Geographical Maps.
"""

import json
import os
from typing import Any, Dict, List
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hmm_viterbi.viterbi_decoder import HMMConfig, ViterbiRouteDecoder

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"
sns.set_theme(style="whitegrid", palette="deep")


def generate_eda_plots(df_inv: pd.DataFrame, test_cache: List[Dict[str, Any]], fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)

    # 1. 4-Panel Dataset Inventory & Quality Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    sns.histplot(
        data=df_inv, x="n_gps", hue="partition", bins=25, kde=True, ax=axes[0, 0], palette=["#1f77b4", "#ff7f0e"]
    )
    axes[0, 0].set_title("(A) Distribution of GPS Observations per Trip (.track)")
    axes[0, 0].set_xlabel("Number of GPS Points (T)")
    axes[0, 0].set_ylabel("Trip Count")

    sns.scatterplot(
        data=df_inv, x="n_nodes", y="n_arcs", hue="partition", style="partition", s=70, ax=axes[0, 1], palette=["#1f77b4", "#ff7f0e"]
    )
    axes[0, 1].set_title("(B) Road Network Complexity (.nodes vs .arcs per Trip)")
    axes[0, 1].set_xlabel("Number of Road Nodes")
    axes[0, 1].set_ylabel("Number of Directed Road Segments (Arcs)")

    sns.boxplot(
        data=df_inv, x="partition", y="n_gaps_gt_1s", ax=axes[1, 0], palette=["#4c72b0", "#dd8452"]
    )
    axes[1, 0].set_title("(C) Irregular Timestamp Gaps (dt > 1.0s) per Trip")
    axes[1, 0].set_xlabel("Dataset Partition")
    axes[1, 0].set_ylabel("Count of Irregular Time Steps (dt > 1s)")

    sc = axes[1, 1].scatter(
        df_inv["ref_lon"], df_inv["ref_lat"], c=df_inv["lon_scale_m_per_deg"], cmap="viridis", s=60, edgecolors="k", linewidth=0.5
    )
    plt.colorbar(sc, ax=axes[1, 1], label="Lon Scale (m/deg)")
    axes[1, 1].set_title("(D) Global Trip Locations & WGS-84 Longitude Scale Variation")
    axes[1, 1].set_xlabel("Reference Longitude (deg)")
    axes[1, 1].set_ylabel("Reference Latitude (deg)")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_dataset_distributions.png"))
    plt.close(fig)

    # 2. GPS Trajectory & Road Network Static Visualization on Selected Trips
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_i, tdata in enumerate(test_cache[:2]):
        ax = axes[ax_i]
        nodes = tdata["nodes"]
        arcs = tdata["arcs"]
        track = tdata["track"]
        route = tdata["route"]

        # Plot road network subset near GPS bounding box
        lon_min, lon_max = track[:, 0].min() - 0.002, track[:, 0].max() + 0.002
        lat_min, lat_max = track[:, 1].min() - 0.002, track[:, 1].max() + 0.002
        u_coords = nodes[arcs[:, 0]]
        v_coords = nodes[arcs[:, 1]]
        mask = (
            (u_coords[:, 0] >= lon_min) & (u_coords[:, 0] <= lon_max) &
            (u_coords[:, 1] >= lat_min) & (u_coords[:, 1] <= lat_max)
        )
        sub_u = u_coords[mask]
        sub_v = v_coords[mask]
        for i in range(min(len(sub_u), 2500)):
            ax.plot([sub_u[i, 0], sub_v[i, 0]], [sub_u[i, 1], sub_v[i, 1]], color="#cccccc", lw=0.6, zorder=1)

        # Ground-truth route
        ru = nodes[arcs[route, 0]]
        rv = nodes[arcs[route, 1]]
        for i in range(len(route)):
            ax.plot(
                [ru[i, 0], rv[i, 0]], [ru[i, 1], rv[i, 1]],
                color="#2ca02c", lw=2.2, zorder=2, label="Reference Route (.route)" if i == 0 else None
            )

        # Noisy GPS track
        ax.scatter(
            track[::4, 0], track[::4, 1], color="#1f77b4", s=10, alpha=0.75, zorder=3, label="GPS Observations (.track)"
        )
        ax.set_title(f"Trip {tdata['trip_id']} ({tdata['partition']}): GPS vs. Road Network")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_trajectory_and_network_static.png"))
    plt.close(fig)


def generate_model_comparison_plots(
    df_comp: pd.DataFrame,
    detailed_results: Dict[str, Any],
    fig_dir: str,
) -> None:
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Core Performance Metrics Bar Chart across All 6 Models
    fig, ax = plt.subplots(figsize=(13, 6))
    metrics_to_plot = [
        "CV Macro F1-score",
        "Student-Test Macro F1-score",
        "Top-1 Accuracy",
        "Route continuity",
    ]
    x = np.arange(len(df_comp))
    width = 0.20
    colors = ["#2b5c8f", "#388e3c", "#d95f02", "#7570b3"]

    for i, col in enumerate(metrics_to_plot):
        vals = df_comp[col].values.astype(float)
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=col, color=colors[i])
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, rotation=0
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df_comp["Model"], rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.50, 1.03)
    ax.set_ylabel("Score")
    ax.set_title("Comparison of 6 Models Across Algorithm Families (5-Fold CV & Student-Test)")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_comparison_metrics.png"))
    plt.close(fig)

    # 2. Log-Loss and Computational Efficiency Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ll_vals = [float(v) for v in df_comp["Log Loss"]]
    bars_ll = axes[0].bar(df_comp["Model"], ll_vals, color="#c44e52", width=0.55)
    for b in bars_ll:
        h = b.get_height()
        axes[0].annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
    axes[0].set_title("Student-Test Log Loss Across Models (Lower is Better)")
    axes[0].set_ylabel("Binary Cross-Entropy Log Loss")
    axes[0].tick_params(axis="x", rotation=20)

    x_m = np.arange(len(df_comp))
    axes[1].bar(x_m - 0.18, df_comp["Training Time (s)"], width=0.35, label="Training Time (s)", color="#4c72b0")
    axes[1].bar(x_m + 0.18, df_comp["Inference Time (s)"], width=0.35, label="Student-Test Inference Time (s)", color="#55a868")
    axes[1].set_xticks(x_m)
    axes[1].set_xticklabels(df_comp["Model"], rotation=20, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Time in Seconds (log scale)")
    axes[1].set_title("Computational Efficiency: Training vs. Inference Time")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_comparison_logloss_and_time.png"))
    plt.close(fig)

    # 3. Confusion Matrices Grid for All 6 Models
    model_names = list(detailed_results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes_flat = axes.flatten()
    for idx, m_name in enumerate(model_names):
        cm = np.array(detailed_results[m_name]["confusion_matrix"])
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        annot = np.empty_like(cm, dtype=object)
        for r in range(2):
            for c in range(2):
                annot[r, c] = f"{cm[r, c]:,}\n({cm_norm[r, c]*100:.1f}%)"
        sns.heatmap(
            cm_norm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=axes_flat[idx],
            xticklabels=["Pred Neg (0)", "Pred Pos (1)"],
            yticklabels=["True Neg (0)", "True Pos (1)"],
        )
        axes_flat[idx].set_title(f"{m_name}\n(Macro F1: {detailed_results[m_name]['summary']['Student-Test Macro F1-score']:.4f})")
    plt.suptitle("Student-Test Confusion Matrices Across All 6 Experimented Models", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrices_all_models.png"))
    plt.close(fig)

    # 4. Epoch-wise Training & Validation Curve ONLY for Deep MLP Classifier
    mlp_info = detailed_results.get("Deep MLP Classifier", {}).get("mlp_history")
    if mlp_info and mlp_info.get("loss_curve"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        epochs = np.arange(1, len(mlp_info["loss_curve"]) + 1)
        axes[0].plot(epochs, mlp_info["loss_curve"], marker="o", color="#1f77b4", lw=2, label="Training Cross-Entropy Loss")
        axes[0].set_title("Deep MLP: Epoch-wise Training Loss Curve")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].legend()

        if mlp_info.get("validation_scores"):
            val_ep = np.arange(1, len(mlp_info["validation_scores"]) + 1)
            axes[1].plot(val_ep, mlp_info["validation_scores"], marker="s", color="#2ca02c", lw=2, label="Validation Accuracy")
            axes[1].set_title("Deep MLP: Epoch-wise Holdout Validation Score")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Validation Accuracy")
            axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "mlp_training_validation_curves.png"))
        plt.close(fig)


def generate_optimisation_and_hmm_plots(
    df_trials: pd.DataFrame,
    df_imp: pd.DataFrame,
    df_per_trip: pd.DataFrame,
    opt_summary: Dict[str, Any],
    fig_dir: str,
) -> None:
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Bayesian Hyperparameter Optimisation History & Untuned vs Tuned Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    trials = df_trials["trial"].values
    f1s = df_trials["cv_macro_f1"].values
    best_so_far = np.maximum.accumulate(f1s)

    axes[0].scatter(trials, f1s, color="#1f77b4", s=55, label="Trial CV Macro F1")
    axes[0].plot(trials, best_so_far, color="#d62728", lw=2.2, label="Best-So-Far TPE Curve")
    axes[0].set_title("Bayesian Optimisation (Optuna TPE) Search Trajectory")
    axes[0].set_xlabel("Optimisation Trial")
    axes[0].set_ylabel("5-Fold GroupKFold Macro F1-score")
    axes[0].legend()

    uvt = opt_summary["untuned_vs_tuned"]
    comp_labels = ["CV Macro F1", "Test Top-1 Acc", "Route Match Acc", "Test Macro F1", "Route Continuity"]
    un_vals = [
        uvt["untuned"]["cv_macro_f1"],
        uvt["untuned"]["student_test_top1_accuracy"],
        uvt["untuned"]["student_test_route_match_accuracy"],
        uvt["untuned"]["student_test_macro_f1"],
        uvt["untuned"]["student_test_route_continuity"],
    ]
    tu_vals = [
        uvt["tuned"]["cv_macro_f1"],
        uvt["tuned"]["student_test_top1_accuracy"],
        uvt["tuned"]["student_test_route_match_accuracy"],
        uvt["tuned"]["student_test_macro_f1"],
        uvt["tuned"]["student_test_route_continuity"],
    ]
    x = np.arange(len(comp_labels))
    axes[1].bar(x - 0.18, un_vals, width=0.35, label="Untuned LightGBM", color="#8c96c6")
    axes[1].bar(x + 0.18, tu_vals, width=0.35, label="Bayesian-Tuned LightGBM", color="#2ca02c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comp_labels, rotation=15, ha="right")
    axes[1].set_ylim(0.80, 1.0)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Untuned vs. Bayesian-Tuned LightGBM Performance")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "bayesian_optimisation_history.png"))
    plt.close(fig)

    # 2. Feature Importance Plot
    fig, ax = plt.subplots(figsize=(10, 6.5))
    top_imp = df_imp.head(15).iloc[::-1]
    ax.barh(top_imp["feature"], top_imp["importance_gain"], color="#2b5c8f")
    ax.set_title("Top 15 Engineered Features by LightGBM Information Gain")
    ax.set_xlabel("Total Split Gain")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "feature_importance_plot.png"))
    plt.close(fig)

    # 3. Classifier-Only vs Classifier + HMM/Viterbi Decoding Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x_t = np.arange(len(df_per_trip))
    axes[0].bar(x_t - 0.18, df_per_trip["clf_top1_accuracy"], width=0.35, label="Classifier-Only Top-1", color="#ff7f0e")
    axes[0].bar(x_t + 0.18, df_per_trip["hmm_top1_accuracy"], width=0.35, label="Classifier + HMM/Viterbi Top-1", color="#1f77b4")
    axes[0].set_xticks(x_t)
    axes[0].set_xticklabels(df_per_trip["trip_id"], rotation=30, ha="right")
    axes[0].set_ylim(0.70, 1.01)
    axes[0].set_ylabel("Observation-Level Top-1 Accuracy")
    axes[0].set_title("Per-Trip Observation Top-1 Accuracy: Classifier vs. HMM+Viterbi")
    axes[0].legend()

    axes[1].bar(x_t - 0.18, df_per_trip["clf_route_continuity"], width=0.35, label="Classifier-Only Continuity", color="#e377c2")
    axes[1].bar(x_t + 0.18, df_per_trip["hmm_route_continuity"], width=0.35, label="Classifier + HMM/Viterbi Continuity", color="#2ca02c")
    axes[1].set_xticks(x_t)
    axes[1].set_xticklabels(df_per_trip["trip_id"], rotation=30, ha="right")
    axes[1].set_ylim(0.80, 1.02)
    axes[1].set_ylabel("Route Continuity")
    axes[1].set_title("Per-Trip Route Continuity: Classifier vs. HMM+Viterbi")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hmm_vs_classifier_comparison.png"))
    plt.close(fig)


def generate_folium_map_for_trip(
    tdata: Dict[str, Any],
    probs_t: np.ndarray,
    decoder: ViterbiRouteDecoder,
    maps_dir: str,
    fig_dir: str,
) -> str:
    """
    Creates an interactive Folium HTML map with LayerControl and custom legend
    showing:
      1. Relevant road network geometry (gray)
      2. Original noisy GPS observations (blue dots)
      3. Ground-truth reference route (green line)
      4. Classifier-only predictions (orange dashed)
      5. Final PathSnap HMM+Viterbi predicted route (crimson line)
    """
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    trip_id = tdata["trip_id"]
    track = tdata["track"]          # (T, 3) [lon, lat, ts]
    nodes = tdata["nodes"]          # (N, 2) [lon, lat]
    arcs = tdata["arcs"]            # (A, 2)
    route = tdata["route"]          # (M,)
    cand_arcs = tdata["cand_arcs"]  # (T, K)
    T = len(track)

    clf_idx = np.argmax(probs_t, axis=1)
    clf_pred_arcs = cand_arcs[np.arange(T), clf_idx]

    dec_out = decoder.decode(
        arcs=arcs,
        nodes_xy=tdata["nodes_xy"],
        cand_arcs=cand_arcs,
        probs=probs_t,
        perp_dist=tdata["perp_dist"],
        proj_frac=tdata["proj_frac"],
        cos_blend=tdata["cos_blend"],
        dt_prev=tdata["dt_prev"],
    )
    hmm_pred_arcs = dec_out["obs_pred_arcs"]
    rec_route = dec_out["reconstructed_route"]

    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    # Layer 1: Local road network segments within trajectory bounding box
    fg_net = folium.FeatureGroup(name="Road Network Geometry (.nodes / .arcs)", show=True)
    lon_min, lon_max = track[:, 0].min() - 0.0015, track[:, 0].max() + 0.0015
    lat_min, lat_max = track[:, 1].min() - 0.0015, track[:, 1].max() + 0.0015
    u_lonlat = nodes[arcs[:, 0]]
    v_lonlat = nodes[arcs[:, 1]]
    bbox_mask = (
        (u_lonlat[:, 0] >= lon_min) & (u_lonlat[:, 0] <= lon_max) &
        (u_lonlat[:, 1] >= lat_min) & (u_lonlat[:, 1] <= lat_max)
    )
    net_indices = np.where(bbox_mask)[0]
    step_net = max(1, len(net_indices) // 1200)
    for a_idx in net_indices[::step_net]:
        u_pt = u_lonlat[a_idx]
        v_pt = v_lonlat[a_idx]
        folium.PolyLine(
            locations=[[float(u_pt[1]), float(u_pt[0])], [float(v_pt[1]), float(v_pt[0])]],
            color="#888888", weight=1.5, opacity=0.45,
        ).add_to(fg_net)
    fg_net.add_to(fmap)

    # Layer 2: Reference Ground-Truth Route (Green)
    if route is not None and len(route) > 0:
        fg_gt = folium.FeatureGroup(name="Reference Ground-Truth Route (.route)", show=True)
        for r_arc in route:
            u_pt = nodes[arcs[int(r_arc), 0]]
            v_pt = nodes[arcs[int(r_arc), 1]]
            folium.PolyLine(
                locations=[[float(u_pt[1]), float(u_pt[0])], [float(v_pt[1]), float(v_pt[0])]],
                color="#1b9e77", weight=6.0, opacity=0.80,
                tooltip=f"Reference Arc {int(r_arc)}",
            ).add_to(fg_gt)
        fg_gt.add_to(fmap)

    # Layer 3: Classifier-Only Predicted Segments (Orange Dashed)
    fg_clf = folium.FeatureGroup(name="Classifier-Only Predictions (No HMM)", show=False)
    clf_unique_seq = [int(clf_pred_arcs[0])]
    for a in clf_pred_arcs[1:]:
        if int(a) != clf_unique_seq[-1]:
            clf_unique_seq.append(int(a))
    for c_arc in clf_unique_seq:
        u_pt = nodes[arcs[c_arc, 0]]
        v_pt = nodes[arcs[c_arc, 1]]
        folium.PolyLine(
            locations=[[float(u_pt[1]), float(u_pt[0])], [float(v_pt[1]), float(v_pt[0])]],
            color="#ff7f00", weight=4.0, opacity=0.85, dash_array="6, 6",
            tooltip=f"Classifier-Only Arc {c_arc}",
        ).add_to(fg_clf)
    fg_clf.add_to(fmap)

    # Layer 4: PathSnap Final Predicted Route (HMM + Viterbi, Red/Crimson)
    fg_pred = folium.FeatureGroup(name="Predicted Route (Tuned LightGBM + HMM/Viterbi)", show=True)
    for p_arc in rec_route:
        u_pt = nodes[arcs[int(p_arc), 0]]
        v_pt = nodes[arcs[int(p_arc), 1]]
        folium.PolyLine(
            locations=[[float(u_pt[1]), float(u_pt[0])], [float(v_pt[1]), float(v_pt[0])]],
            color="#e41a1c", weight=3.5, opacity=0.92,
            tooltip=f"Predicted Route Arc {int(p_arc)}",
        ).add_to(fg_pred)
    fg_pred.add_to(fmap)

    # Layer 5: Raw GPS Observations (Blue Circle Markers)
    fg_gps = folium.FeatureGroup(name="Original Noisy GPS Observations (.track)", show=True)
    gps_step = max(1, T // 350)
    for t_i in range(0, T, gps_step):
        lon_i, lat_i, ts_i = track[t_i]
        folium.CircleMarker(
            location=[float(lat_i), float(lon_i)],
            radius=3.2,
            color="#0055cc",
            fill=True,
            fill_color="#3388ff",
            fill_opacity=0.85,
            tooltip=f"t={int(ts_i)}s | Pred Arc={int(hmm_pred_arcs[t_i])}",
        ).add_to(fg_gps)

    # Start and End Markers
    folium.Marker(
        location=[float(track[0, 1]), float(track[0, 0])],
        popup=f"Trip {trip_id} Start (t={track[0, 2]:.0f}s)",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fg_gps)
    folium.Marker(
        location=[float(track[-1, 1]), float(track[-1, 0])],
        popup=f"Trip {trip_id} End (t={track[-1, 2]:.0f}s)",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(fg_gps)
    fg_gps.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[float(track[:, 1].min()), float(track[:, 0].min())], [float(track[:, 1].max()), float(track[:, 0].max())]])

    # Custom HTML Legend
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background-color: white;
                padding: 12px 16px; border: 2px solid #444; border-radius: 8px; font-family: Arial, sans-serif;
                font-size: 13px; box-shadow: 2px 2px 6px rgba(0,0,0,0.25);">
        <b>PathSnap Map-Matching — Trip {trip_id}</b><br>
        <span style="color:#0055cc;">&#9679;</span> Original GPS Observations (.track)<br>
        <span style="color:#888888;">&#9473;&#9473;</span> Road Network Segments (.arcs)<br>
        <span style="color:#1b9e77;">&#9644;&#9644;</span> Reference Ground-Truth Route (.route)<br>
        <span style="color:#e41a1c;">&#9473;&#9473;</span> Predicted Route (LightGBM + HMM/Viterbi)<br>
        <span style="color:#ff7f00;">&#9476;&#9476;</span> Classifier-Only Predictions (Toggle in Layers)
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    html_path = os.path.join(maps_dir, f"trip_{trip_id}_folium_map.html")
    fmap.save(html_path)
    return html_path


def generate_static_map_comparison_figure(
    test_cache: List[Dict[str, Any]],
    tuned_probs_test: List[np.ndarray],
    decoder: ViterbiRouteDecoder,
    fig_dir: str,
) -> None:
    """
    Creates a high-resolution 2x2 static map comparison view (for the PDF report)
    comparing Raw GPS, Road Network, Reference Route, and Predicted HMM+Viterbi Route.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes_flat = axes.flatten()

    selected_indices = [0, 4, 6, 9]  # Trips 00000000, 00000052, 00000065, 00000091
    for plot_i, idx in enumerate(selected_indices):
        ax = axes_flat[plot_i]
        tdata = test_cache[idx]
        probs_t = tuned_probs_test[idx]
        trip_id = tdata["trip_id"]
        track = tdata["track"]
        nodes = tdata["nodes"]
        arcs = tdata["arcs"]
        route = tdata["route"]
        cand_arcs = tdata["cand_arcs"]
        gt_arcs = tdata["gt_arcs"]

        dec_out = decoder.decode(
            arcs=arcs,
            nodes_xy=tdata["nodes_xy"],
            cand_arcs=cand_arcs,
            probs=probs_t,
            perp_dist=tdata["perp_dist"],
            proj_frac=tdata["proj_frac"],
            cos_blend=tdata["cos_blend"],
            dt_prev=tdata["dt_prev"],
        )
        rec_route = dec_out["reconstructed_route"]
        acc_vit = float(np.mean(dec_out["obs_pred_arcs"] == gt_arcs))

        lon_min, lon_max = track[:, 0].min() - 0.001, track[:, 0].max() + 0.001
        lat_min, lat_max = track[:, 1].min() - 0.001, track[:, 1].max() + 0.001
        u_coords = nodes[arcs[:, 0]]
        v_coords = nodes[arcs[:, 1]]
        mask = (
            (u_coords[:, 0] >= lon_min) & (u_coords[:, 0] <= lon_max) &
            (u_coords[:, 1] >= lat_min) & (u_coords[:, 1] <= lat_max)
        )
        sub_u = u_coords[mask]
        sub_v = v_coords[mask]
        for i in range(min(len(sub_u), 2000)):
            ax.plot([sub_u[i, 0], sub_v[i, 0]], [sub_u[i, 1], sub_v[i, 1]], color="#d3d3d3", lw=0.6, zorder=1)

        # Reference route (thick green)
        ru = nodes[arcs[route, 0]]
        rv = nodes[arcs[route, 1]]
        for i in range(len(route)):
            ax.plot(
                [ru[i, 0], rv[i, 0]], [ru[i, 1], rv[i, 1]],
                color="#2ca02c", lw=3.8, alpha=0.65, zorder=2, label="Reference Route (.route)" if i == 0 else None
            )

        # Predicted HMM+Viterbi route (dashed crimson overlay)
        pu = nodes[arcs[rec_route, 0]]
        pv = nodes[arcs[rec_route, 1]]
        for i in range(len(rec_route)):
            ax.plot(
                [pu[i, 0], pv[i, 0]], [pu[i, 1], pv[i, 1]],
                color="#d62728", lw=1.8, ls="--", zorder=3, label="Predicted Route (HMM+Viterbi)" if i == 0 else None
            )

        # Raw GPS observations (blue dots)
        ax.scatter(
            track[::3, 0], track[::3, 1], color="#1f77b4", s=8, alpha=0.75, zorder=4, label="Noisy GPS (.track)"
        )
        ax.set_title(f"Student-Test Trip {trip_id} | Top-1 Acc: {acc_vit*100:.2f}%")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.legend(loc="best", fontsize=8)

    plt.suptitle("PathSnap Map-Matching: Reference Route vs. Predicted HMM+Viterbi Route & Raw GPS", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "folium_map_comparison_view.png"))
    plt.close(fig)
