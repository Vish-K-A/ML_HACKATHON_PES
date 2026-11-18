"""HMM / Viterbi experiments and final artifact export.

Usage: python scripts/05_hmm_experiments.py [MODEL]   (default LightGBM)

1. Out-of-fold (5-fold GroupKFold) scores of the tuned model on the training
   trips, so the decoder is tuned on honest, unseen-trip probabilities.
2. Grid search of decoder parameters (beta, emission_power) on a seeded subset
   of 30 training trips -> chosen by mean observation accuracy.
3. Student-test comparison: classifier-only vs classifier + HMM (pointwise,
   continuity, route-level precision/recall/F1, breaks) per trip.
4. Gap robustness: tracks downsampled to 1 point / 5 s, 15 s and 30 s.
5. Folium maps for every student-test trip.
6. Export models/pathsnap_final.joblib (model + features + HMM params + config).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.config import HMM_DEFAULTS, MODELS_DIR, RESULTS_DIR, SEED
from pathsnap.features import FEATURES, build_context, build_features
from pathsnap.hmm import reconstruct_route, viterbi
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.labels import align_route
from pathsnap.metrics import continuity, route_overlap
from pathsnap.viz import trip_map

OUT = RESULTS_DIR / "hmm"
OUT.mkdir(parents=True, exist_ok=True)
GRID_BETA = [2.0, 5.0, 10.0, 20.0, 50.0]
GRID_W = [0.5, 1.0, 2.0]


def split_scores(ctx, s):
    sizes = [len(c[0]) for c in ctx.cands]
    return np.split(np.asarray(s), np.cumsum(sizes)[:-1])


def evaluate_trip(ctx, per_point, ref_route, params, ref_arc=None):
    c = ctx.net.canon
    if ref_arc is None:
        ref_arc, _ = align_route(ctx.xy, ref_route, ctx.net)
    clf = np.array([cc[0][np.argmax(s)] for cc, s in zip(ctx.cands, per_point)])
    t0 = time.perf_counter()
    _, hmm, info = viterbi(ctx, per_point, params)
    vt = time.perf_counter() - t0
    r_h = reconstruct_route(ctx.net, hmm)
    r_c = reconstruct_route(ctx.net, clf)
    oh = route_overlap(r_h, ref_route, ctx.net.length, c)
    oc = route_overlap(r_c, ref_route, ctx.net.length, c)
    return {"n_points": len(clf),
            "clf_acc": np.mean(c[clf] == c[ref_arc]), "hmm_acc": np.mean(c[hmm] == c[ref_arc]),
            "clf_continuity": continuity(clf, ctx.net), "hmm_continuity": continuity(hmm, ctx.net),
            "clf_route_precision": oc["route_precision"], "clf_route_recall": oc["route_recall"], "clf_route_f1": oc["route_f1"],
            "hmm_route_precision": oh["route_precision"], "hmm_route_recall": oh["route_recall"], "hmm_route_f1": oh["route_f1"],
            "n_breaks": info["n_breaks"], "viterbi_s": vt}, clf, hmm, r_h


def main(model_name="LightGBM"):
    art = joblib.load(MODELS_DIR / f"{model_name}_tuned.joblib")
    model, params_model = art["model"], art["params"]
    from importlib import import_module
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    bo = import_module("04_bayes_opt")
    make = bo.SPACES[model_name]["make"]

    # ---------------------------------------------------------- 1. OOF scores
    tr = pd.read_parquet(RESULTS_DIR / "features_train.parquet")
    pts_tr = pd.read_parquet(RESULTS_DIR / "points_train.parquet")
    oof_path = OUT / f"oof_scores_{model_name}.npy"
    if oof_path.exists():
        oof = np.load(oof_path)
    else:
        hitkeys = pts_tr.loc[pts_tr.cand_hit == 1, ["trip", "point"]].assign(_h=1)
        trm = tr.merge(hitkeys, on=["trip", "point"], how="left")
        oof = np.zeros(len(tr), dtype=np.float32)
        for k, (itr, iva) in enumerate(GroupKFold(5).split(tr, tr.label, groups=tr.trip)):
            fit_rows = trm.iloc[itr]
            fit_rows = fit_rows[fit_rows._h == 1]
            est = make(params_model)
            est.fit(fit_rows[FEATURES].to_numpy(), fit_rows.label.to_numpy())
            oof[iva] = est.predict_proba(tr.iloc[iva][FEATURES].to_numpy())[:, 1]
            print("oof fold", k)
        np.save(oof_path, oof)
    tr["s"] = oof
    # per-trip out-of-fold Top-1 (shows which training trips are hard / badly labelled)
    from pathsnap.metrics import top1_predictions
    top = top1_predictions(tr, oof)
    pt = pts_tr.groupby("trip").agg(n_points=("point", "size"), cand_hit=("cand_hit", "mean"),
                                    align_dist_median=("align_dist", "median"))
    pt["oof_top1_all_points"] = top.groupby("trip").label.sum() / pt.n_points
    pt["oof_top1_hit_points"] = top.groupby("trip").label.sum() / (pt.cand_hit * pt.n_points)
    pt.sort_values("oof_top1_all_points").to_csv(OUT / f"oof_per_trip_{model_name}.csv")

    # ---------------------------------------------------------- 2. decoder grid
    rng = np.random.default_rng(SEED)
    trips = list_trips("train")
    sub = sorted(rng.choice(trips, 30, replace=False).tolist())
    grid_rows = []
    for tid in sub:
        ctx = build_context(load_trip("train", tid))
        g = tr[tr.trip == tid]
        per_point = split_scores(ctx, g.s.to_numpy())
        ref_route = load_route("train", tid)
        ref_arc, _ = align_route(ctx.xy, ref_route, ctx.net)
        for b in GRID_BETA:
            for w in GRID_W:
                m, *_ = evaluate_trip(ctx, per_point, ref_route, {"beta": b, "emission_power": w}, ref_arc)
                grid_rows.append({"trip": tid, "beta": b, "emission_power": w, **m})
        print("grid", tid)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "decoder_grid_train_trips.csv", index=False)
    agg = grid.groupby(["beta", "emission_power"]).apply(
        lambda d: pd.Series({"hmm_acc": np.average(d.hmm_acc, weights=d.n_points),
                             "clf_acc": np.average(d.clf_acc, weights=d.n_points),
                             "hmm_route_f1": d.hmm_route_f1.mean(), "hmm_continuity": d.hmm_continuity.mean()}),
        include_groups=False).reset_index()
    agg.to_csv(OUT / "decoder_grid_summary.csv", index=False)
    best = agg.sort_values("hmm_acc", ascending=False).iloc[0]
    hmm_params = {**HMM_DEFAULTS, "beta": float(best.beta), "emission_power": float(best.emission_power)}
    print("decoder best", hmm_params)
    piv = agg.pivot(index="beta", columns="emission_power", values="hmm_acc")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(piv.values, cmap="viridis")
    for (i, j), v in np.ndenumerate(piv.values):
        ax.text(j, i, f"{v:.4f}", ha="center", va="center", color="w", fontsize=8)
    ax.set(xticks=range(len(piv.columns)), xticklabels=piv.columns, yticks=range(len(piv.index)),
           yticklabels=piv.index, xlabel="emission power w", ylabel="beta (m)",
           title="HMM accuracy on 30 training trips (OOF scores)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT / "decoder_grid.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------- 3. student test
    te = pd.read_parquet(RESULTS_DIR / "features_student_test.parquet")
    te["s"] = model.predict_proba(te[FEATURES].to_numpy())[:, 1]
    rows, maps_dir = [], RESULTS_DIR / "maps"
    for tid in list_trips("student_test"):
        trip = load_trip("student_test", tid)
        ctx = build_context(trip)
        per_point = split_scores(ctx, te[te.trip == tid].s.to_numpy())
        ref = load_route("student_test", tid)
        m, clf, hmm, route = evaluate_trip(ctx, per_point, ref, hmm_params)
        rows.append({"trip": tid, **m})
        tm = trip_map(trip, reference=ref, classifier=reconstruct_route(ctx.net, clf), hmm=route,
                      title=f"Trip {tid}: classifier {m['clf_acc']:.3f} / HMM {m['hmm_acc']:.3f} accuracy")
        tm.save(str(maps_dir / f"student_test_{tid}.html"))
        print("test", tid, round(m["clf_acc"], 4), round(m["hmm_acc"], 4))
    res = pd.DataFrame(rows)
    w = res.n_points
    tot = {"trip": "ALL (point-weighted)", "n_points": int(w.sum())}
    for c in res.columns.drop(["trip", "n_points"]):
        tot[c] = np.average(res[c], weights=w) if c not in ("n_breaks", "viterbi_s") else res[c].sum()
    res = pd.concat([res, pd.DataFrame([tot])], ignore_index=True)
    res.to_csv(OUT / "student_test_classifier_vs_hmm.csv", index=False)
    print(res.tail(1).T)

    r = res.iloc[:-1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(r)); wd = .38
    for a, (k1, k2, ttl) in zip(ax, [("clf_acc", "hmm_acc", "Pointwise accuracy"),
                                     ("clf_continuity", "hmm_continuity", "Route continuity"),
                                     ("clf_route_f1", "hmm_route_f1", "Route F1 (length-weighted)")]):
        a.bar(x - wd / 2, r[k1], wd, label="classifier only"); a.bar(x + wd / 2, r[k2], wd, label="classifier + HMM")
        a.set_xticks(x, r.trip.str[-2:]); a.set(title=ttl, xlabel="student-test trip (last 2 digits)")
        a.set_ylim(max(0, min(r[k1].min(), r[k2].min()) - .05), 1.005)
    ax[0].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "classifier_vs_hmm_student_test.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------- 4. gap robustness
    gap_rows = []
    for step in [5, 15, 30]:
        for tid in list_trips("student_test"):
            trip = load_trip("student_test", tid)
            t = trip.track[:, 2]
            keep = np.r_[True, np.diff(np.floor((t - t[0]) / step)) > 0]
            trip_ds = replace(trip, track=trip.track[keep])
            ctx = build_context(trip_ds)
            df = build_features(ctx)
            per_point = split_scores(ctx, model.predict_proba(df[FEATURES].to_numpy())[:, 1])
            m, *_ = evaluate_trip(ctx, per_point, load_route("student_test", tid), hmm_params)
            gap_rows.append({"sampling_s": step, "trip": tid, **m})
    gap = pd.DataFrame(gap_rows)
    gs = gap.groupby("sampling_s").apply(lambda d: pd.Series({
        "points": d.n_points.sum(),
        "clf_acc": np.average(d.clf_acc, weights=d.n_points), "hmm_acc": np.average(d.hmm_acc, weights=d.n_points),
        "clf_route_f1": d.clf_route_f1.mean(), "hmm_route_f1": d.hmm_route_f1.mean(),
        "clf_continuity": d.clf_continuity.mean(), "hmm_continuity": d.hmm_continuity.mean()}), include_groups=False)
    full = res.iloc[-1]
    gs.loc[1] = [full.n_points, full.clf_acc, full.hmm_acc, r.clf_route_f1.mean(), r.hmm_route_f1.mean(),
                 r.clf_continuity.mean(), r.hmm_continuity.mean()]
    gs = gs.sort_index()
    gs.to_csv(OUT / "gap_robustness.csv")
    gap.to_csv(OUT / "gap_robustness_per_trip.csv", index=False)
    print(gs)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, lab in [("clf_acc", "classifier only: accuracy"), ("hmm_acc", "HMM: accuracy"),
                   ("clf_route_f1", "classifier only: route F1"), ("hmm_route_f1", "HMM: route F1")]:
        ax.plot(gs.index, gs[k], marker="o", label=lab, ls="-" if "hmm" in k else "--")
    ax.set(xlabel="sampling interval (s)", ylabel="score", title="Effect of sparser GPS (student test)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "gap_robustness.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------- 6. export
    artifact = {"model": model, "model_name": model_name, "model_params": params_model,
                "features": FEATURES, "hmm_params": hmm_params,
                "candidate_config": {"radius_m": config.CAND_RADIUS_M, "k": config.CAND_K,
                                     "densify_step_m": config.DENSIFY_STEP_M, "margin_m": config.NETWORK_MARGIN_M},
                "trained_on": "train partition (80 trips), observations with reference arc in candidate set",
                "seed": SEED, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    joblib.dump(artifact, MODELS_DIR / "pathsnap_final.joblib", compress=3)
    (MODELS_DIR / "pathsnap_final_config.json").write_text(json.dumps(
        {k: v for k, v in artifact.items() if k != "model"}, indent=2, default=str))
    print("exported", MODELS_DIR / "pathsnap_final.joblib")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "LightGBM")
