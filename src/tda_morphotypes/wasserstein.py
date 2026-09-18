"""Exact Wasserstein distances between persistence diagrams (equation 1 of the paper).

The optimal partial matching is solved as a linear assignment problem on the
diagrams augmented with their projections onto the diagonal. Values agree with
``gudhi.wasserstein.wasserstein_distance`` (see tests) and are several times
faster to compute on the small diagrams of this study.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .parallel import parallel_map, shared


def _ground_cost(X: np.ndarray, Y: np.ndarray, order: float, internal_p: float) -> np.ndarray:
    if np.isinf(internal_p):
        return cdist(X, Y, metric="chebyshev") ** order
    return cdist(X, Y, metric="minkowski", p=internal_p) ** order


def _diagonal_cost(X: np.ndarray, order: float, internal_p: float) -> np.ndarray:
    # distance, for the internal_p-norm, from (b, d) to its projection onto the diagonal
    return ((X[:, 1] - X[:, 0]) * 2.0 ** (1.0 / internal_p - 1.0)) ** order


def _essential_cost(X: np.ndarray, Y: np.ndarray, order: float) -> float:
    """Cost of matching the points with infinite coordinates (same type only)."""
    cost = 0.0
    for kind in ((False, True), (True, False), (True, True)):  # (-inf birth, +inf death)
        sx = X[(np.isneginf(X[:, 0]) == kind[0]) & (np.isposinf(X[:, 1]) == kind[1])]
        sy = Y[(np.isneginf(Y[:, 0]) == kind[0]) & (np.isposinf(Y[:, 1]) == kind[1])]
        if len(sx) != len(sy):
            return np.inf
        if kind == (True, True) or not len(sx):
            continue
        coord = 1 if kind[0] else 0  # the finite coordinate
        cost += np.sum(np.abs(np.sort(sx[:, coord]) - np.sort(sy[:, coord])) ** order)
    return cost


def wasserstein_distance(X, Y, order: float = 2.0, internal_p: float = 2.0) -> float:
    """(order, internal_p)-Wasserstein distance between two persistence diagrams."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 2)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 2)
    fx, fy = np.isfinite(X).all(axis=1), np.isfinite(Y).all(axis=1)
    cost = _essential_cost(X[~fx], Y[~fy], order) if (~fx).any() or (~fy).any() else 0.0
    if np.isinf(cost):
        return np.inf
    X, Y = X[fx], Y[fy]
    n, m = len(X), len(Y)
    if n + m:
        C = np.zeros((n + m, n + m))
        C[:n, :m] = _ground_cost(X, Y, order, internal_p)
        C[:n, m:] = _diagonal_cost(X, order, internal_p)[:, None]
        C[n:, :m] = _diagonal_cost(Y, order, internal_p)[None, :]
        rows, cols = linear_sum_assignment(C)
        cost += C[rows, cols].sum()
    return cost ** (1.0 / order)


def decolored_dist(X1, X2, dim=None, p: float = 2.0, order: float = 2.0):
    """Distance between two decolored diagrams ``[H0, H1, H2]``.

    With ``dim=None``: (sum_d W_d^p)^(1/p) over the three degrees, W_d being
    the (order, p)-Wasserstein distance. With a list of degrees ``dim``: the
    list of the (p, p)-Wasserstein distances of these degrees.
    """
    if dim is None:
        return sum(wasserstein_distance(X1[d], X2[d], order, p) ** p for d in range(3)) ** (1.0 / p)
    return [wasserstein_distance(X1[d], X2[d], p, p) for d in dim]


def _row_distances(i: int) -> np.ndarray:
    diagrams, order, internal_p = shared()
    return np.array([wasserstein_distance(diagrams[i], diagrams[j], order, internal_p)
                     for j in range(i + 1, len(diagrams))])


def distance_matrix(diagrams, order: float = 2.0, internal_p: float = 2.0, n_jobs: int = 1,
                    desc: str = "") -> np.ndarray:
    """Distances between all pairs of diagrams of one degree, as a condensed matrix.

    The entries are in the order of ``scipy.spatial.distance`` (pairs (i, j),
    i < j, sorted), i.e. the ``distances`` obtained in the original notebooks
    from ``zip(*sorted(dict_dist.items()))``.
    """
    n = len(diagrams)
    # pair long and short rows so that the chunks sent to the workers are balanced
    rows = [i for pair in zip(range(n // 2), range(n - 1, n // 2 - 1, -1)) for i in pair]
    if n % 2:
        rows.append(n // 2)
    values = parallel_map(_row_distances, rows, n_jobs=n_jobs, desc=desc,
                          shared=(list(diagrams), order, internal_p), chunksize=8)
    by_row = dict(zip(rows, values))
    return np.concatenate([by_row[i] for i in range(n)]) if n > 1 else np.empty(0)


def combine(per_degree: dict[int, np.ndarray], degrees, p: float = 2.0) -> np.ndarray:
    """l_p combination of per-degree distances: (sum_d W_d^p)^(1/p)."""
    return sum(np.power(per_degree[d], p) for d in degrees) ** (1.0 / p)
