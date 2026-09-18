"""Figures of the paper."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

from .geometry import _ROT_COS, _ROT_SIN  # noqa: E402

DEGREE_COLORS = {0: "tab:red", 1: "tab:blue", 2: "tab:green"}
METHOD_LABELS = {"complete": "Complete", "ward": "Ward", "kmeans": "K-Means", "kmedoids": "K-Medoids"}
METHOD_COLORS = {"complete": "tab:gray", "ward": "tab:orange", "kmeans": "tab:green", "kmedoids": "tab:red"}


def _save(fig, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _gdi_axes(ax) -> None:
    ax.set_xlim(1.5, 30.5)
    ax.set_xticks(range(2, 31, 2))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel(r"$K$ (number of clusters)")
    ax.set_ylabel("GDI score")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="lower right")


def plot_gdi_curves(curves, representation: str, path) -> None:
    """GDI as a function of the number of clusters (Figures 13 and 14)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = curves[curves["representation"] == representation]
    for method, group in sub.groupby("method", sort=False):
        ax.plot(group["k"], group["gdi"], ".-", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    _gdi_axes(ax)
    _save(fig, path)


def plot_gdi_body_vs_trunk(curves, representation: str, path) -> None:
    """Whole body (solid) against trunk (dashed) (Figures 16 and 17)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for part, style in (("body", "-"), ("trunk", "--")):
        sub = curves[curves["representation"] == f"{representation}_{part}"]
        for method, group in sub.groupby("method", sort=False):
            ax.plot(group["k"], group["gdi"], style, color=METHOD_COLORS[method],
                    label=f"{METHOD_LABELS[method]} - {part}")
    _gdi_axes(ax)
    _save(fig, path)


def plot_dendrogram(Z, n_leaves: int, path, labels=None, ylabel: str = "") -> None:
    """Dendrogram truncated to its last ``n_leaves`` clusters (Figures 7, 11, 18, 21).

    Clusters reduced to one individual are labelled by its name when ``labels``
    is given, the others by their size in parentheses.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=n_leaves, show_contracted=True,
               labels=labels, color_threshold=0, above_threshold_color="k", leaf_rotation=90)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    _save(fig, path)


def plot_anomaly_percentages(percentages, best, path) -> None:
    """percent1 (blue), percent2 (red) and their mean percent3 (yellow) of section 3."""
    fig, ax = plt.subplots(figsize=(8, 5))
    k = percentages["nb_clus"]
    ax.plot(k, percentages["percent1"], color="tab:blue", label="solos that are anomalies")
    ax.plot(k, percentages["percent2"], color="tab:red", label="anomalies that are solos")
    ax.plot(k, percentages["percent3"], color="gold", label="mean (criterion)", linewidth=2)
    ax.axvspan(best[0], best[1], color="0.85", zorder=0, label=f"best range [{best[0]}, {best[1]}]")
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("%")
    ax.set_ylim(-2, 102)
    ax.legend()
    _save(fig, path)


def plot_indices(table, n_selected: int, path) -> None:
    """Elbow, Davies-Bouldin, Silhouette and Dunn indices against the number of clusters."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    titles = {"merge_height": "Elbow (Ward merge height)", "davies_bouldin": "Davies-Bouldin (min)",
              "silhouette": "Silhouette (max)", "dunn": "Dunn (max)"}
    for ax, (column, title) in zip(axes.ravel(), titles.items()):
        ax.plot(table["n_clusters"], table[column], "o-", markersize=3)
        ax.axvline(n_selected, color="tab:red", linestyle="--")
        ax.set_title(title)
        ax.grid(True, alpha=0.4)
    for ax in axes[1]:
        ax.set_xlabel("number of clusters")
    fig.tight_layout()
    _save(fig, path)


def plot_diagram_and_barcode(classes, path, highlight=()) -> None:
    """Persistence diagram and barcode (Figures 2 and 9); bars are numbered as in the paper."""
    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
    finite = [c.death for c in classes if np.isfinite(c.death)]
    top = max(finite) * 1.1 if finite else 1.0
    for c in classes:
        death = c.death if np.isfinite(c.death) else top
        ax_d.scatter(c.birth, death, color=DEGREE_COLORS[c.degree], s=40 if c.number in highlight else 15,
                     edgecolor="k" if c.number in highlight else "none", zorder=3)
        ax_b.barh(c.number, death - c.birth, left=c.birth, color=DEGREE_COLORS[c.degree], height=0.8,
                  edgecolor="k" if c.number in highlight else "none")
    ax_d.plot([0, top], [0, top], "k", linewidth=0.8)
    ax_d.axhline(top, color="k", linestyle=":", linewidth=0.8)
    ax_d.set_xlim(-0.02 * top, top * 1.02)
    ax_d.set_ylim(-0.02 * top, top * 1.05)
    ax_d.set_xlabel("Birth")
    ax_d.set_ylabel("Death")
    ax_d.set_title("Persistence diagram")
    for d, color in DEGREE_COLORS.items():
        ax_d.scatter([], [], color=color, label=f"H{d}")
    ax_d.legend(loc="lower right")
    ax_b.set_yticks([c.number for c in classes])
    ax_b.set_xlabel("Filtration value (squared radius)")
    ax_b.set_title("Persistence barcode")
    ax_b.invert_yaxis()
    _save(fig, path)


# --- front views of bodies --------------------------------------------------------
# Once rotated by _front, the SPRING individuals face +y; they are drawn as seen
# from the front with an orthographic projection (image axes: -x, z).


def _front(points: np.ndarray, centre: np.ndarray | None = None) -> np.ndarray:
    """Rotate a SPRING point cloud (about ``centre``) so that the shoulders lie along x."""
    p = np.asarray(points, dtype=float) - (points.mean(axis=0) if centre is None else centre)
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0] = x * _ROT_COS + y * _ROT_SIN
    p[:, 1] = y * _ROT_COS - x * _ROT_SIN
    return p


def _image(p: np.ndarray) -> np.ndarray:
    """Front projection of points given in the frame of ``_front``."""
    return np.column_stack([-p[:, 0], p[:, 2]])


def _below_neck(points: np.ndarray, fraction: float | None) -> np.ndarray:
    z = points[:, 2]
    if fraction is None:
        return np.ones(len(z), dtype=bool)
    return z <= z.min() + fraction * (z.max() - z.min())


def _panels(n_panels: int, ncols: int, panel=(2.0, 3.6)):
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.35 / (panel[1] * nrows),
                        wspace=0.02, hspace=0.12)
    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.set_axis_off()
    for ax in axes.ravel()[n_panels:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def plot_H0(ax, c, img) -> None:
    """H0 class: the vertex whose component dies."""
    ax.scatter(*img(c.birth_simplex).T, s=25, color="tab:green")


def plot_H1(ax, c, img) -> None:
    """H1 class: loop at its birth (green) and triangle killing it (red)."""
    ax.plot(*img(c.loop).T, color="tab:green", linewidth=1.5)
    if c.death_simplex is not None:
        ax.add_patch(Polygon(img(c.death_simplex), color="tab:red", alpha=0.5))


def plot_H2(ax, c, img) -> None:
    """H2 class: triangle creating it (green) and tetrahedron killing it (red)."""
    ax.add_patch(Polygon(img(c.birth_simplex), color="tab:green", alpha=0.8))
    if c.death_simplex is not None:
        tetra = img(c.death_simplex)
        for face in combinations(range(4), 3):
            ax.add_patch(Polygon(tetra[list(face)], color="tab:red", alpha=0.5))


def plot_homology_classes(points: np.ndarray, classes, path, ncols: int = 4, hide_head: float | None = 0.87,
                          by_degree: bool = False) -> None:
    """Representatives of homology classes on the point cloud (Figures 5 and 10).

    Panels are titled with the barcode number, or with the number within the
    degree if ``by_degree``.
    """
    centre = points.mean(axis=0)

    def img(q):
        return _image(_front(q, centre))

    body = img(points[_below_neck(points, hide_head)])
    fig, axes = _panels(len(classes), ncols)
    for ax, c in zip(axes, classes):
        ax.scatter(*body.T, s=0.05, color="0.7", linewidths=0)
        (plot_H0, plot_H1, plot_H2)[c.degree](ax, c, img)
        ax.set_title(f"H{c.degree} n°{c.degree_number}" if by_degree else f"n°{c.number}: H{c.degree}",
                     fontsize=10)
        ax.autoscale_view()
    _save(fig, path)


_LIGHT = np.array([-0.3, 1.0, 0.4]) / np.linalg.norm([-0.3, 1.0, 0.4])


def render_meshes(meshes, path, ncols: int | None = None, hide_head: float | None = 0.87) -> None:
    """Front renders of meshes given as (points, faces, title) (Figures 1, 6, 8, 12, 19, 20, 22).

    As in the paper, the heads are cut off (above ``hide_head`` times the height).
    """
    ncols = ncols or len(meshes)
    fig, axes = _panels(len(meshes), ncols)
    for ax, (points, faces, title) in zip(axes, meshes):
        faces = faces[_below_neck(points, hide_head)[faces].all(axis=1)]
        tri = _front(points)[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        shade = 0.25 + 0.7 * np.abs(normals @ _LIGHT)
        order = np.argsort(tri[:, :, 1].mean(axis=1))  # painter's algorithm: far faces first
        ax.add_collection(PolyCollection(np.stack([-tri[order, :, 0], tri[order, :, 2]], axis=-1),
                                         facecolors=np.repeat(shade[order, None], 3, axis=1),
                                         edgecolors="face", linewidths=0.2))
        ax.autoscale_view()
        ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_trunk(body: np.ndarray, trunk: np.ndarray, path, hide_head: float | None = 0.87) -> None:
    """A body point cloud and its isolated trunk (Figure 15); ``trunk`` is in the frame of body_to_trunk."""
    fig, axes = _panels(2, 2, panel=(2.6, 4.0))
    shown = _image(_front(body[_below_neck(body, hide_head)], body.mean(axis=0)))
    for ax, cloud, title in zip(axes, (shown, _image(trunk)), ("(a) Full body", "(b) Trunk")):
        ax.scatter(*cloud.T, s=0.1, color="tab:blue", linewidths=0)
        ax.set_title(title, fontsize=10)
    _save(fig, path)
