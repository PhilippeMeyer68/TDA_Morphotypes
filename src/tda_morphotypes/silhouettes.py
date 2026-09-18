"""Persistence silhouettes (Chazal et al., 2014) and the vectors of section 2.2.

``Silhouette`` has the interface and the behaviour of
``gudhi.representations.Silhouette`` in gudhi 3.5/3.6, used for the paper:

* the sampling range is [min birth, max death] over the diagrams given to
  ``fit`` (newer versions of gudhi shrink it by half a step unless
  ``keep_endpoints=True``);
* an empty diagram gives a zero vector (newer versions return NaN).

The silhouette of a diagram is the weighted mean of the tent functions
t -> max(0, min(t - b, d - t)) of its points, multiplied by sqrt(2).
"""

from __future__ import annotations

import numpy as np

from .config import SilhouetteSetting

# Weight functions of a diagram point x = (birth, death), as in the original notebooks.
WEIGHTS = {
    "uniform": lambda x: 1,
    "birth": lambda x: x[0],
    "ratio": lambda x: (x[1] + x[0]) / (x[1] - x[0]),
}


class Silhouette:
    def __init__(self, resolution: int = 100, weight=lambda x: 1, sample_range=(np.nan, np.nan)):
        self.resolution = resolution
        self.weight = WEIGHTS[weight] if isinstance(weight, str) else weight
        self.sample_range = sample_range

    def fit(self, X, y=None) -> "Silhouette":
        """Fit the sampling range on a list of diagrams (unless given)."""
        nonempty = [D for D in X if len(D)]
        if np.isnan(self.sample_range).any() and nonempty:
            P = np.concatenate(nonempty)
            self.sample_range = np.where(np.isnan(self.sample_range), [P[:, 0].min(), P[:, 1].max()],
                                         self.sample_range)
        return self

    def transform(self, X) -> np.ndarray:
        """Silhouettes of a list of finite diagrams; shape (len(X), resolution)."""
        if np.isnan(self.sample_range).any():  # nothing to sample: all diagrams are empty
            return np.zeros((len(X), self.resolution))
        grid = np.linspace(self.sample_range[0], self.sample_range[1], self.resolution)
        out = np.zeros((len(X), self.resolution))
        for i, D in enumerate(X):
            if len(D):
                w = np.array([self.weight(x) for x in D], dtype=np.float64)
                tents = np.maximum(0.0, np.minimum(grid[:, None] - D[None, :, 0], D[None, :, 1] - grid[:, None]))
                out[i] = np.sqrt(2) * tents @ (w / w.sum())
        return out

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X).transform(X)


def silhouette_vectors(X_decolor, setting: SilhouetteSetting, standardize: bool = True) -> np.ndarray:
    """Concatenated H0, H1 and H2 silhouettes of every scan (525 values in the paper).

    ``X_decolor`` is a DiagramSet or a list of decolored diagrams. The essential
    H0 class is left out. Degrees listed in ``setting.zero_degrees`` give a
    block of zeros. The vectors are standardised by the mean and standard
    deviation of all their entries, as in the original notebooks.
    """
    diagrams = X_decolor.diagrams if hasattr(X_decolor, "diagrams") else X_decolor
    blocks = []
    for d, (resolution, weight) in enumerate(zip(setting.resolutions, setting.weights)):
        test = [np.asarray(xd[d]).reshape(-1, 2) for xd in diagrams]
        test = [D[np.isfinite(D).all(axis=1)] for D in test]
        if d in setting.zero_degrees:
            blocks.append(np.zeros((len(test), resolution)))
        else:
            blocks.append(Silhouette(resolution=resolution, weight=weight).fit_transform(test))
    sh = np.hstack(blocks)
    if standardize:
        sh = (sh - sh.mean()) / sh.std()
    return sh
