"""Geographic helpers.

Trips are spread across the globe, so each trip is projected to its own UTM
zone (chosen from the track centroid). Inside a zone UTM distortion is < 0.1 %,
which is far below GPS noise, so planar Euclidean distance in metres is used
for all geometry afterwards.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


class Projector:
    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg(lon0, lat0)
        self._fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)

    def __call__(self, lonlat: np.ndarray) -> np.ndarray:
        x, y = self._fwd.transform(lonlat[:, 0], lonlat[:, 1])
        return np.c_[x, y]


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised projection of points p onto segments a->b (all (n,2)).

    Returns (distance, t) where t in [0,1] is the clipped position along the
    segment and distance is to the clipped projection.
    """
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.einsum("ij,ij->i", p - a, ab) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return np.hypot(*(p - proj).T), t


def bearing(dx, dy):
    """Planar bearing in radians, 0 = north, clockwise."""
    return np.arctan2(dx, dy)


def angle_diff(a, b):
    """Absolute smallest difference between two angles (radians), in [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)
