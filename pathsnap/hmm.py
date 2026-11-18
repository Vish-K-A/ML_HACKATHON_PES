"""HMM / Viterbi route reconstruction.

Hidden state  : (observation i, candidate arc c, position t on c). The state
                list for observation i is exactly the classifier's candidate
                list, so candidate <-> arc correspondence is preserved.
Emission      : log P(z_i | c) ~ w * log q_ic, where q_ic is the classifier
                probability normalised over the candidates of observation i
                (w = emission_power).
Transition    : Newson & Krumm (2009) style. With d_gps = straight-line distance
                between consecutive observations and d_route = network distance
                between the two projected positions (directed graph, bounded
                Dijkstra), log P = -|d_route - d_gps| / beta. Small backward
                moves on the same arc (GPS jitter while stopped) are allowed at
                their along-arc length. Unreachable pairs get -inf.
Breaks        : if dt > gap_break_s, or no candidate of i is reachable from any
                candidate of i-1, the chain is split and Viterbi restarts from
                the emission scores. Every observation still receives an arc.
"""
from __future__ import annotations

import numpy as np

from .config import HMM_DEFAULTS

NEG = -1e18
BACKWARD_TOL_M = 30.0


def viterbi(ctx, scores_per_point: list[np.ndarray], params: dict | None = None):
    """Decode the most likely candidate sequence.

    scores_per_point[i] is an array of classifier probabilities aligned with
    ctx.cands[i]. Returns (chosen index per point, chosen arc per point, info).
    """
    P = {**HMM_DEFAULTS, **(params or {})}
    beta, w = P["beta"], P["emission_power"]
    net, xy, t = ctx.net, ctx.xy, ctx.track[:, 2]
    n = len(xy)
    cands = ctx.cands
    choice = np.zeros(n, dtype=np.int64)
    breaks = []

    def emis(i):
        s = np.asarray(scores_per_point[i], dtype=float)
        q = s / max(s.sum(), 1e-12)
        return w * np.log(np.clip(q, 1e-12, None))

    delta = emis(0)
    back: list[np.ndarray | None] = [None]
    seg_end: dict[int, np.ndarray] = {}   # final delta of a segment ending at i
    for i in range(1, n):
        a_arcs, _, a_t = cands[i - 1]
        b_arcs, _, b_t = cands[i]
        dt = t[i] - t[i - 1]
        e = emis(i)
        if dt > P["gap_break_s"] or len(a_arcs) == 0 or len(b_arcs) == 0:
            back.append(None); breaks.append((i, "gap" if dt > P["gap_break_s"] else "empty"))
            seg_end[i - 1] = delta; delta = e; continue
        dg = float(np.hypot(*(xy[i] - xy[i - 1])))
        limit = max(P["min_search_m"], P["max_speed_mps"] * dt, 2.5 * dg)
        T = np.full((len(a_arcs), len(b_arcs)), NEG)
        for ja, (aa, ta) in enumerate(zip(a_arcs, a_t)):
            if delta[ja] <= NEG / 2:
                continue
            for jb, (bb, tb) in enumerate(zip(b_arcs, b_t)):
                if aa == bb and tb < ta:
                    back_m = (ta - tb) * net.length[aa]
                    dr = back_m if back_m <= BACKWARD_TOL_M else np.inf
                else:
                    dr = net.route_distance(int(aa), float(ta), int(bb), float(tb), limit)
                if np.isfinite(dr):
                    T[ja, jb] = -abs(dr - dg) / beta
        tot = delta[:, None] + T
        bp = np.argmax(tot, axis=0)
        best = tot[bp, np.arange(len(b_arcs))]
        if np.all(best <= NEG / 2):
            back.append(None); breaks.append((i, "unreachable"))
            seg_end[i - 1] = delta; delta = e; continue
        back.append(bp)
        delta = best + e

    # backtrack segment by segment (a segment ends where back[i] is None)
    j = int(np.argmax(delta))
    for i in range(n - 1, -1, -1):
        choice[i] = j
        if i == 0:
            break
        bp = back[i]
        j = int(np.argmax(seg_end[i - 1])) if bp is None else int(bp[j])
    arcs = np.array([cands[i][0][choice[i]] if len(cands[i][0]) else -1 for i in range(n)])
    return choice, arcs, {"breaks": breaks, "n_breaks": len(breaks)}


def reconstruct_route(net, point_arcs: np.ndarray, max_fill_m: float = 3000.0) -> list[int]:
    """Turn per-observation arcs into a traversed arc sequence, filling the
    shortest path between consecutive distinct arcs when they are not adjacent."""
    from .metrics import collapse
    seq = [a for a in collapse(point_arcs) if a >= 0]
    out: list[int] = []
    for a in seq:
        if out and not net.connected(out[-1], a):
            out.extend(net.path_arcs(out[-1], a, limit=max_fill_m))
        if not out or out[-1] != a:
            out.append(a)
    return out
