"""Point-cloud preprocessing: height normalisation (section 2.4) and trunk isolation (section 4.2)."""

from __future__ import annotations

import numpy as np

from .config import DiagramSetting


def scan_height(scan: np.ndarray) -> float:
    """Height of a scan: extent of the point cloud along the vertical axis z."""
    z = scan[:, 2]
    return float(z.max() - z.min())


def normalize_scan(scan: np.ndarray, size: float = 170.0, decimals: int | None = None) -> np.ndarray:
    """Homothety bringing the height of the individual to ``size``.

    If ``decimals`` is given, the measured height is rounded first, as in the
    original computations of sections 3 and 5 (``size=1.7, decimals=2``).
    """
    height = scan_height(scan)
    if decimals is not None:
        height = np.round(height, decimals)
    return scan * (size / height)


# Horizontal rotation aligning the shoulders of the SPRING meshes with the x axis.
_ROT_COS = -0.5734623443633283
_ROT_SIN = 0.8192319205190405


def body_to_trunk(points: np.ndarray) -> np.ndarray:
    """Isolate the trunk of a body point cloud (section 4.2, Figure 15).

    The body is translated to stand on z = 0, scaled to 170 cm and rotated so
    that the shoulders lie along x. The 10990 lowest vertices (the head is
    dropped) are restricted to an 80 cm high band below the highest of them,
    then points are kept between two pairs of lines of the (x, z) plane,
    symmetric with respect to x = 0, whose slopes depend on the width of the
    individual.

    The constants are those of the original implementation; the vertices of
    the SPRING meshes are in correspondence, so index-based steps select the
    same body parts for every individual.
    """
    points = points[np.argsort(points[:, 2], kind="stable")]
    origin = points.mean(axis=0)
    origin[2] = points[0, 2]
    to_plot = (points - origin).T
    to_plot *= 170 / to_plot[2, -1]
    to_plot[0] -= 0.7

    to_plot = to_plot[:, :10990]
    temp = to_plot.copy()
    to_plot[0] = temp[0] * _ROT_COS + temp[1] * _ROT_SIN
    to_plot[1] = temp[1] * _ROT_COS - temp[0] * _ROT_SIN

    crotch = to_plot[2, -1] - 80
    x = np.max(np.abs(to_plot[0, 10400:]))  # half width at shoulder level
    to_plot = to_plot[:, to_plot[2] > crotch]
    to_plot[2] -= np.min(to_plot[2])

    X = np.abs(to_plot[0])
    a = x * (1 + (70 - to_plot[2]) * 0.001)
    temp = to_plot[:, X < a]
    temp = temp[:, temp[2] < 30]
    b = np.max(temp[0] ** 2 + temp[1] ** 2) ** 0.5  # half width at hip level

    a = (40 - to_plot[2]) * (b - x) / 30 + b
    keep = (X < a) | ((X < b) & (to_plot[1] < 40))
    return to_plot[:, keep].T


def prepare_points(points: np.ndarray, setting: DiagramSetting) -> np.ndarray:
    """Point cloud on which the diagrams of ``setting`` are computed."""
    if setting.trunk:
        return body_to_trunk(points)
    return normalize_scan(points, setting.target_height, setting.height_decimals)
