"""Readers and geometry helpers for the whitespace-delimited PathSnap data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray  # longitude, latitude, timestamp
    nodes: np.ndarray  # longitude, latitude
    arcs: np.ndarray  # from-node, to-node; row index is arc ID
    route: np.ndarray  # ordered arc IDs


def _load(path: Path, columns: int | None = None) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if columns is not None and values.shape[1] != columns:
        raise ValueError(f"{path} should have {columns} columns; got {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite value found in {path}")
    return values


def load_trip(folder: Path) -> Trip:
    trip_id = folder.name
    trip = Trip(
        trip_id=trip_id,
        track=_load(folder / f"{trip_id}.track", 3),
        nodes=_load(folder / f"{trip_id}.nodes", 2),
        arcs=_load(folder / f"{trip_id}.arcs", 2).astype(np.int64),
        route=_load(folder / f"{trip_id}.route", 1).reshape(-1).astype(np.int64),
    )
    if trip.track.shape[0] < 2 or trip.route.size == 0:
        raise ValueError(f"Trip {trip_id} has too few track points or no route arcs")
    if trip.arcs.min() < 0 or trip.arcs.max() >= trip.nodes.shape[0]:
        raise ValueError(f"Trip {trip_id}: arc endpoint outside node array")
    if trip.route.min() < 0 or trip.route.max() >= trip.arcs.shape[0]:
        raise ValueError(f"Trip {trip_id}: route arc ID outside arc array")
    return trip


def project_trip(trip: Trip) -> tuple[np.ndarray, np.ndarray]:
    """Project all trip coordinates to the local UTM zone, in metres."""
    lon0 = float(np.nanmedian(trip.nodes[:, 0]))
    lat0 = float(np.nanmedian(trip.nodes[:, 1]))
    zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    node_x, node_y = transformer.transform(trip.nodes[:, 0], trip.nodes[:, 1])
    track_x, track_y = transformer.transform(trip.track[:, 0], trip.track[:, 1])
    return np.column_stack((node_x, node_y)), np.column_stack((track_x, track_y))
