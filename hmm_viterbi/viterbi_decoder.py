"""
Hidden Markov Model (HMM) and Viterbi Route Reconstruction for PathSnap.
Combines ML classifier candidate probabilities with road-network graph topology,
multi-hop reachability, and gap-aware transition costs.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class HMMConfig:
    clf_weight: float = 5.5          # Weight on negative log-probability from ML classifier
    geom_dist_weight: float = 0.25   # Auxiliary spatial emission regularization
    geom_head_weight: float = 1.20   # Auxiliary directional emission regularization
    stay_cost: float = 0.0           # Transition cost for staying on the same road segment
    backward_proj_penalty: float = 2.0  # Penalty if projection moves backwards along same segment
    hop1_cost: float = 0.30          # Direct connected transition (v_prev == u_curr)
    hop2_cost: float = 1.50          # 2-hop connected transition
    hop3_cost: float = 3.20          # 3-hop connected transition
    uturn_cost: float = 11.50        # Immediate reversal (u->v followed by v->u)
    disconnected_base_cost: float = 8.50  # Base penalty for disconnected transition
    disconnected_dist_weight: float = 0.25  # Additional penalty per meter of node gap / dt


class ViterbiRouteDecoder:
    """
    Sequential route decoder using an HMM formulation and Viterbi dynamic programming.
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()

    @staticmethod
    def build_adjacency(arcs: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
        """Maps each node u -> list of (arc_id, v_node)."""
        out_adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for arc_id, (u, v) in enumerate(arcs):
            out_adj[int(u)].append((int(arc_id), int(v)))
        return out_adj

    def compute_emission_costs(
        self,
        probs: np.ndarray,
        perp_dist: np.ndarray,
        cos_blend: np.ndarray,
    ) -> np.ndarray:
        """
        Converts ML classifier probabilities P(y=1 | x_{t,k}) and geometric cues
        into negative log-emission costs for each (timestamp t, candidate k).
        """
        cfg = self.config
        clipped_p = np.clip(probs, 1e-5, 1.0 - 1e-6)
        neg_log_p = -np.log(clipped_p)
        geom_term = cfg.geom_dist_weight * perp_dist + cfg.geom_head_weight * (1.0 - cos_blend)
        return cfg.clf_weight * neg_log_p + geom_term

    def decode(
        self,
        arcs: np.ndarray,
        nodes_xy: np.ndarray,
        cand_arcs: np.ndarray,
        probs: np.ndarray,
        perp_dist: np.ndarray,
        proj_frac: np.ndarray,
        cos_blend: np.ndarray,
        dt_prev: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Runs the Viterbi algorithm over the candidate lattice (T, K).
        Returns:
            obs_pred_arcs: (T,) predicted road segment (arc_id) for each GPS observation
            obs_pred_idx: (T,) chosen candidate index in 0..K-1
            reconstructed_route: (R,) continuous sequence of road segments bridging multi-hop transitions
        """
        cfg = self.config
        T, K = cand_arcs.shape
        out_adj = self.build_adjacency(arcs)

        emit_cost = self.compute_emission_costs(probs, perp_dist, cos_blend)
        dp = np.full((T, K), np.inf, dtype=np.float64)
        back = np.zeros((T, K), dtype=np.int32)
        dp[0] = emit_cost[0]

        # Cache 2-hop and 3-hop reachability per node for fast transition evaluation
        hop2_cache: Dict[int, set] = {}
        hop3_cache: Dict[int, set] = {}

        def get_hops(node_v: int) -> Tuple[set, set]:
            if node_v not in hop2_cache:
                h2 = set()
                h3 = set()
                for _, n1 in out_adj.get(node_v, ()):
                    h2.add(n1)
                    for _, n2 in out_adj.get(n1, ()):
                        h3.add(n2)
                hop2_cache[node_v] = h2
                hop3_cache[node_v] = h3
            return hop2_cache[node_v], hop3_cache[node_v]

        for t in range(1, T):
            prev_a = cand_arcs[t - 1]
            curr_a = cand_arcs[t]
            prev_u = arcs[prev_a, 0]
            prev_v = arcs[prev_a, 1]
            curr_u = arcs[curr_a, 0]
            curr_v = arcs[curr_a, 1]
            dt = max(1.0, float(dt_prev[t]))

            trans = np.empty((K, K), dtype=np.float64)
            for p_i in range(K):
                pa = int(prev_a[p_i])
                pu = int(prev_u[p_i])
                pv = int(prev_v[p_i])
                ptau = float(proj_frac[t - 1, p_i])
                h2, h3 = get_hops(pv)
                pv_xy = nodes_xy[pv]

                for c_i in range(K):
                    ca = int(curr_a[c_i])
                    cu = int(curr_u[c_i])
                    cv = int(curr_v[c_i])
                    ctau = float(proj_frac[t, c_i])

                    if pa == ca:
                        # Same segment: penalize slightly if projection moved noticeably backward
                        back_creep = max(0.0, ptau - ctau - 0.12)
                        trans[p_i, c_i] = cfg.stay_cost + cfg.backward_proj_penalty * back_creep
                    elif pv == cu and cv != pu:
                        # Direct 1-hop connected segment
                        trans[p_i, c_i] = cfg.hop1_cost
                    elif cu in h2 and cv != pu:
                        # 2-hop connected segment
                        trans[p_i, c_i] = cfg.hop2_cost / min(dt, 2.0)
                    elif cu in h3 and cv != pu:
                        # 3-hop connected segment
                        trans[p_i, c_i] = cfg.hop3_cost / min(dt, 3.0)
                    elif pv == cu and cv == pu:
                        # Immediate U-turn onto reverse edge
                        trans[p_i, c_i] = cfg.uturn_cost
                    else:
                        # Disconnected segments or larger GPS gap
                        gap_dist = float(np.linalg.norm(pv_xy - nodes_xy[cu]))
                        gap_cost = cfg.disconnected_base_cost + cfg.disconnected_dist_weight * (gap_dist / dt)
                        trans[p_i, c_i] = min(22.0, gap_cost)

            total = dp[t - 1, :, None] + trans  # (K_prev, K_curr)
            back[t] = np.argmin(total, axis=0)
            dp[t] = emit_cost[t] + np.min(total, axis=0)

        best_idx = np.zeros(T, dtype=np.int32)
        best_idx[-1] = int(np.argmin(dp[-1]))
        for t in range(T - 1, 0, -1):
            best_idx[t - 1] = back[t, best_idx[t]]

        obs_pred_arcs = cand_arcs[np.arange(T), best_idx]
        reconstructed_route = self.reconstruct_continuous_route(arcs, obs_pred_arcs, out_adj)

        return {
            "obs_pred_arcs": obs_pred_arcs,
            "obs_pred_idx": best_idx,
            "reconstructed_route": reconstructed_route,
        }

    @staticmethod
    def reconstruct_continuous_route(
        arcs: np.ndarray,
        obs_pred_arcs: np.ndarray,
        out_adj: Dict[int, List[Tuple[int, int]]],
        max_bfs_hops: int = 5,
    ) -> np.ndarray:
        """
        Converts observation-level predicted road segments into a deduplicated,
        topologically bridged route sequence (filling 1..max_bfs_hops intermediate arcs via BFS).
        """
        if len(obs_pred_arcs) == 0:
            return np.array([], dtype=np.int64)

        collapsed: List[int] = [int(obs_pred_arcs[0])]
        for a in obs_pred_arcs[1:]:
            ia = int(a)
            if ia != collapsed[-1]:
                collapsed.append(ia)

        full_route: List[int] = [collapsed[0]]
        for next_arc in collapsed[1:]:
            prev_arc = full_route[-1]
            v_prev = int(arcs[prev_arc, 1])
            u_next = int(arcs[next_arc, 0])

            if v_prev == u_next:
                full_route.append(next_arc)
                continue

            # Short BFS to bridge intermediate skipped segments (e.g. during 1-5s GPS gaps)
            queue = deque([(v_prev, [])])
            visited = {v_prev}
            bridge: Optional[List[int]] = None
            while queue:
                curr_node, path_arcs = queue.popleft()
                if len(path_arcs) >= max_bfs_hops:
                    continue
                for edge_id, nbr_node in out_adj.get(curr_node, ()):
                    if nbr_node == u_next:
                        bridge = path_arcs + [edge_id]
                        break
                    if nbr_node not in visited:
                        visited.add(nbr_node)
                        queue.append((nbr_node, path_arcs + [edge_id]))
                if bridge is not None:
                    break

            if bridge is not None:
                full_route.extend(bridge)
            full_route.append(next_arc)

        return np.array(full_route, dtype=np.int64)
