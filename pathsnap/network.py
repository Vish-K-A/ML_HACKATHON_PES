"""Road-network representation for one trip.

* nodes are projected to the trip's UTM zone (metres)
* only arcs touching the track bounding box (+ margin) are indexed for
  candidate search; arc ids are kept as the original 0-based row indices
* duplicate arcs (same from/to pair) are mapped to a canonical id, so
  geometrically identical arcs are not counted as different answers
* a bounded Dijkstra over the directed graph gives network distances used by
  the HMM transition model
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .config import CAND_K, CAND_RADIUS_M, DENSIFY_STEP_M, NETWORK_MARGIN_M
from .geo import Projector, bearing, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: Projector,
                 bbox_xy: tuple[float, float, float, float] | None = None):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        self.arcs = arcs
        self.xy = proj(nodes_lonlat)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        a = self.xy[arcs[:, 0]]
        b = self.xy[arcs[:, 1]]
        self.A, self.B = a, b
        d = b - a
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = bearing(d[:, 0], d[:, 1])
        self.out_deg = np.bincount(arcs[:, 0], minlength=self.n_nodes)
        self.in_deg = np.bincount(arcs[:, 1], minlength=self.n_nodes)

        # canonical id for duplicated (from,to) pairs
        _, first, inv = np.unique(arcs, axis=0, return_index=True, return_inverse=True)
        self.canon = first[inv.ravel()]

        # reverse-arc lookup (is the road two-way?)
        key = arcs[:, 0] * self.n_nodes + arcs[:, 1]
        rkey = arcs[:, 1] * self.n_nodes + arcs[:, 0]
        self.has_reverse = np.isin(rkey, key)

        # adjacency (CSR by from-node) for Dijkstra
        order = np.argsort(arcs[:, 0], kind="stable")
        self._adj_arc = order
        self._adj_ptr = np.searchsorted(arcs[order, 0], np.arange(self.n_nodes + 1))
        self._adj_to = arcs[order, 1]
        self._adj_w = self.length[order]
        self._dcache: dict[int, tuple[float, dict[int, float]]] = {}

        # spatial index over (clipped) arcs
        if bbox_xy is None:
            sel = np.arange(self.n_arcs)
        else:
            x0, y0, x1, y1 = bbox_xy
            m = NETWORK_MARGIN_M
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            sel = np.where((hi[:, 0] >= x0 - m) & (lo[:, 0] <= x1 + m) &
                           (hi[:, 1] >= y0 - m) & (lo[:, 1] <= y1 + m))[0]
        self.indexed_arcs = sel
        n_samp = np.maximum(1, np.ceil(self.length[sel] / DENSIFY_STEP_M).astype(int)) + 1
        arc_rep = np.repeat(sel, n_samp)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n_samp]) if len(sel) else np.empty(0)
        pts = self.A[arc_rep] + (self.B[arc_rep] - self.A[arc_rep]) * frac[:, None]
        self._samp_arc = arc_rep
        self._tree = cKDTree(pts) if len(pts) else None

    # ------------------------------------------------------------ candidates
    def candidates(self, p: np.ndarray, radius: float = CAND_RADIUS_M, k: int = CAND_K):
        """Candidate arcs for each point in p (n,2).

        Returns a list (one entry per point) of (arc_ids, dist, t) arrays sorted
        by distance. Arcs within `radius` are kept (max k); if none, the single
        nearest arc set (up to k within 3*radius, else the nearest) is used so
        every observation has at least one state.
        """
        out = []
        if self._tree is None:
            return [(np.empty(0, int), np.empty(0), np.empty(0)) for _ in p]
        r_q = radius + DENSIFY_STEP_M
        hits = self._tree.query_ball_point(p, r_q)
        for i, h in enumerate(hits):
            if not h:
                _, j = self._tree.query(p[i], k=64)
                h = j[np.isfinite(_)] if np.ndim(j) else [j]
                arcs = np.unique(self._samp_arc[np.asarray(h)])
                d, t = point_segment(np.repeat(p[i][None], len(arcs), 0), self.A[arcs], self.B[arcs])
                o = np.argsort(d)
                keep = o[d[o] <= max(3 * radius, d[o[0]])][:k]
                out.append((arcs[keep], d[keep], t[keep]))
                continue
            arcs = np.unique(self._samp_arc[np.asarray(h)])
            d, t = point_segment(np.repeat(p[i][None], len(arcs), 0), self.A[arcs], self.B[arcs])
            o = np.argsort(d)
            o = o[d[o] <= radius][:k]
            if len(o) == 0:
                o = np.argsort(d)[:1]
            out.append((arcs[o], d[o], t[o]))
        return out

    # ------------------------------------------------------------ routing
    def dist_from(self, src: int, limit: float) -> dict[int, float]:
        """Bounded single-source shortest path distances from node src."""
        cached = self._dcache.get(src)
        if cached is not None and cached[0] >= limit:
            return cached[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        ptr, to, w = self._adj_ptr, self._adj_to, self._adj_w
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for j in range(ptr[u], ptr[u + 1]):
                v = int(to[j])
                nd = d + w[j]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._dcache[src] = (limit, dist)
        return dist

    def route_distance(self, arc_a: int, t_a: float, arc_b: int, t_b: float, limit: float) -> float:
        """Network distance from position t_a on arc_a to t_b on arc_b (inf if > limit)."""
        if arc_a == arc_b and t_b >= t_a - 1e-9:
            return (t_b - t_a) * self.length[arc_a]
        rem = (1.0 - t_a) * self.length[arc_a]
        head = t_b * self.length[arc_b]
        need = limit - rem - head
        if need < 0:
            return np.inf
        d = self.dist_from(int(self.arcs[arc_a, 1]), need).get(int(self.arcs[arc_b, 0]))
        return np.inf if d is None or d > need else rem + d + head

    def connected(self, arc_a: int, arc_b: int) -> bool:
        """True if arc_b can directly follow arc_a (or they are the same road)."""
        return (arc_a == arc_b or self.canon[arc_a] == self.canon[arc_b]
                or self.arcs[arc_a, 1] == self.arcs[arc_b, 0])

    def path_arcs(self, arc_a: int, arc_b: int, limit: float = 5000.0) -> list[int]:
        """Arcs of the shortest path from end of arc_a to start of arc_b (exclusive)."""
        src, dst = int(self.arcs[arc_a, 1]), int(self.arcs[arc_b, 0])
        if src == dst:
            return []
        prev: dict[int, tuple[int, int]] = {}
        dist = {src: 0.0}
        heap = [(0.0, src)]
        ptr, to, w, aid = self._adj_ptr, self._adj_to, self._adj_w, self._adj_arc
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for j in range(ptr[u], ptr[u + 1]):
                v = int(to[j]); nd = d + w[j]
                if nd < dist.get(v, np.inf):
                    dist[v] = nd; prev[v] = (u, int(aid[j]))
                    heapq.heappush(heap, (nd, v))
        if dst not in prev:
            return []
        path, v = [], dst
        while v != src:
            u, a = prev[v]; path.append(a); v = u
        return path[::-1]
