"""Persistence diagrams of alpha complexes.

As in the original code, a *decolored* diagram is the list ``[H0, H1, H2]`` of
the (birth, death) arrays of each homological degree, as opposed to gudhi's
list of ``(degree, (birth, death))`` pairs. The H0 array contains the
essential class (0, inf). Filtration values are squared radii.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gudhi
import numpy as np

from .geometry import normalize_scan, scan_height

MAX_DEGREE = 2


class point_set:
    """A point cloud with its alpha complex, simplex tree and persistence (computed once)."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.array(points, dtype=np.float64)
        self.nb, self.dim = self.points.shape
        self.__alpha_complex = None
        self.__simplex_tree = None
        self.__persistence = None
        self.__min_persistence = None

    def AlphaComplex(self) -> gudhi.AlphaComplex:
        if self.__alpha_complex is None:
            self.__alpha_complex = gudhi.AlphaComplex(points=self.points)
        return self.__alpha_complex

    def SimplexTree(self) -> gudhi.SimplexTree:
        if self.__simplex_tree is None:
            self.__simplex_tree = self.AlphaComplex().create_simplex_tree()
        return self.__simplex_tree

    def Persistence(self, min_persistence: float | None = None) -> list:
        """gudhi's persistence, keeping the intervals longer than ``min_persistence``.

        Without argument, returns the persistence computed last (with threshold 0
        if none was).
        """
        if self.__persistence is None or (min_persistence is not None and min_persistence != self.__min_persistence):
            self.__min_persistence = 0.0 if min_persistence is None else min_persistence
            self.__persistence = self.SimplexTree().persistence(min_persistence=self.__min_persistence)
        return self.__persistence

    def DecoloredPersistence(self, min_persistence: float | None = None) -> list[np.ndarray]:
        """Decolored diagram ``[H0, H1, H2]``, arrays of shape (n, 2)."""
        self.Persistence(min_persistence)
        tree = self.SimplexTree()
        return [np.asarray(tree.persistence_intervals_in_dimension(d), dtype=np.float64).reshape(-1, 2)
                for d in range(MAX_DEGREE + 1)]

    def height(self) -> float:
        return scan_height(self.points)

    def normalize(self, size: float = 170.0, decimals: int | None = None) -> None:
        """Scale the cloud in place so that the individual is ``size`` tall (see ``normalize_scan``)."""
        if self.__alpha_complex is not None:
            raise RuntimeError("normalize before computing the alpha complex")
        self.points = normalize_scan(self.points, size, decimals)


def decolored_diag(points: np.ndarray, min_persistence: float) -> list[np.ndarray]:
    """Decolored diagram of the alpha complex of ``points``."""
    return point_set(points).DecoloredPersistence(min_persistence)


@dataclass
class DiagramSet:
    """Decolored diagrams (``X_decolor``) of a list of scans, with their names."""

    names: list[str]
    diagrams: list[list[np.ndarray]]

    def __post_init__(self):
        if len(self.names) != len(self.diagrams):
            raise ValueError("names and diagrams must have the same length")

    def __len__(self) -> int:
        return len(self.names)

    def degree(self, d: int, finite: bool = False) -> list[np.ndarray]:
        """Diagrams of degree ``d``; ``finite`` drops the essential intervals."""
        out = [dgm[d] for dgm in self.diagrams]
        if finite:
            out = [D[np.isfinite(D).all(axis=1)] for D in out]
        return out

    def filtered(self, min_persistence: float) -> "DiagramSet":
        """Keep the intervals longer than ``min_persistence`` (as gudhi does)."""
        return DiagramSet(
            list(self.names),
            [[D[(D[:, 1] - D[:, 0]) > min_persistence] for D in dgm] for dgm in self.diagrams],
        )

    def subset(self, indices) -> "DiagramSet":
        return DiagramSet([self.names[i] for i in indices], [self.diagrams[i] for i in indices])

    def without(self, names) -> "DiagramSet":
        excluded = set(names)
        return self.subset([i for i, n in enumerate(self.names) if n not in excluded])

    def __add__(self, other: "DiagramSet") -> "DiagramSet":
        return DiagramSet(self.names + other.names, self.diagrams + other.diagrams)

    def save(self, path: str | Path) -> None:
        arrays = {"names": np.array(self.names)}
        for d in range(MAX_DEGREE + 1):
            dgms = self.degree(d)
            arrays[f"counts_{d}"] = np.array([len(D) for D in dgms], dtype=np.int64)
            arrays[f"points_{d}"] = np.concatenate(dgms) if dgms else np.empty((0, 2))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "DiagramSet":
        with np.load(path) as f:
            names = [str(n) for n in f["names"]]
            per_degree = []
            for d in range(MAX_DEGREE + 1):
                bounds = np.cumsum(f[f"counts_{d}"])[:-1]
                per_degree.append(np.split(f[f"points_{d}"], bounds))
        return cls(names, [list(dgm) for dgm in zip(*per_degree)])
