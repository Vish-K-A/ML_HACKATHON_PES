"""Folium map generation (shared by EDA, experiments and the report)."""
from __future__ import annotations

import folium
import numpy as np
from branca.element import MacroElement, Template
from scipy.spatial import cKDTree

from .io import Trip

COLORS = {"gps": "#1f77b4", "network": "#9e9e9e", "reference": "#2ca02c",
          "classifier": "#ff7f0e", "hmm": "#d62728"}


def _arc_lines(trip: Trip, arc_ids) -> list[list[tuple[float, float]]]:
    n = trip.nodes
    a = trip.arcs[np.asarray(arc_ids, dtype=int)]
    return [[(n[f, 1], n[f, 0]), (n[t, 1], n[t, 0])] for f, t in a]


def _merge_route(trip: Trip, arc_ids) -> list[list[tuple[float, float]]]:
    """Join consecutive connected arcs into polylines (fewer map objects)."""
    lines, cur, last_to = [], [], None
    n = trip.nodes
    for a in arc_ids:
        f, t = trip.arcs[int(a)]
        if last_to is not None and f == last_to:
            cur.append((n[t, 1], n[t, 0]))
        else:
            if cur:
                lines.append(cur)
            cur = [(n[f, 1], n[f, 0]), (n[t, 1], n[t, 0])]
        last_to = t
    if cur:
        lines.append(cur)
    return lines


def _legend(items: dict[str, str]) -> MacroElement:
    rows = "".join(f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{k}</div>' for k, c in items.items())
    html = ("{% macro html(this, kwargs) %}<div style='position:fixed;bottom:24px;left:24px;z-index:9999;"
            "background:white;padding:8px 10px;border:1px solid #888;border-radius:6px;font:12px sans-serif'>"
            f"<b>Legend</b>{rows}</div>{{% endmacro %}}")
    el = MacroElement(); el._template = Template(html)
    return el


def trip_map(trip: Trip, reference=None, classifier=None, hmm=None,
             network_buffer_m: float = 150.0, title: str | None = None) -> folium.Map:
    tr = trip.track
    m = folium.Map(location=[np.median(tr[:, 1]), np.median(tr[:, 0])], zoom_start=14,
                   tiles="cartodbpositron", control_scale=True)
    # nearby road network (arcs with an endpoint within buffer of the track)
    lat0 = np.radians(np.median(tr[:, 1]))
    sc = np.array([111320 * np.cos(lat0), 110540])
    tree = cKDTree(tr[:, :2] * sc)
    d, _ = tree.query(trip.nodes * sc, distance_upper_bound=network_buffer_m)
    near = np.isfinite(d)
    arcs = np.where(near[trip.arcs[:, 0]] | near[trip.arcs[:, 1]])[0]
    fg = folium.FeatureGroup(name=f"Road network ({len(arcs)} arcs)")
    for line in _arc_lines(trip, arcs):
        folium.PolyLine(line, color=COLORS["network"], weight=1.5, opacity=.6).add_to(fg)
    fg.add_to(m)
    legend = {"Road network": COLORS["network"]}
    for name, arcs_seq, key, w in (("Reference route", reference, "reference", 7),
                                   ("Classifier only", classifier, "classifier", 4),
                                   ("Classifier + HMM", hmm, "hmm", 4)):
        if arcs_seq is None:
            continue
        fg = folium.FeatureGroup(name=name)
        for line in _merge_route(trip, arcs_seq):
            folium.PolyLine(line, color=COLORS[key], weight=w, opacity=.8).add_to(fg)
        fg.add_to(m); legend[name] = COLORS[key]
    fg = folium.FeatureGroup(name=f"GPS observations ({len(tr)})")
    step = max(1, len(tr) // 2500)
    for i in range(0, len(tr), step):
        folium.CircleMarker((tr[i, 1], tr[i, 0]), radius=2, color=COLORS["gps"], fill=True,
                            weight=0, fill_opacity=.8, tooltip=f"t={tr[i,2]:.0f}s").add_to(fg)
    fg.add_to(m); legend["GPS observation"] = COLORS["gps"]
    folium.Marker((tr[0, 1], tr[0, 0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((tr[-1, 1], tr[-1, 0]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().add_child(_legend(legend))
    if title:
        m.get_root().html.add_child(folium.Element(
            f"<h4 style='position:fixed;top:8px;left:60px;z-index:9999;background:white;padding:4px 8px;"
            f"border-radius:4px;font-family:sans-serif'>{title}</h4>"))
    return m
