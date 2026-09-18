"""Clustering, with the exact semantics of the library versions used for the paper.

* Hierarchical clusterings are done with ``scipy.cluster.hierarchy.linkage``
  and ``fcluster(Z, nb_clus, criterion="maxclust")``, as in the notebooks (on
  a matrix of Wasserstein distances, Ward's method goes through the
  Lance-Williams formula).
* ``KMeans`` behaves like ``sklearn.cluster.KMeans`` 1.0/1.1: ten k-means++
  initialisations drawn one after the other from the same random generator,
  the first centre being drawn with ``randint``. scikit-learn >= 1.3 draws it
  differently, hence the explicit initialisation below.
* ``KMedoids`` is ``sklearn_extra.cluster.KMedoids`` 0.2.0 with
  ``method="pam", init="k-medoids++"``, reimplemented because
  scikit-learn-extra no longer builds on recent Python versions. Swap
  selection and tie-breaking are identical.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans as _SklearnKMeans
from sklearn.metrics import pairwise_distances


def make_dict_clusters(clusters) -> dict[int, list[int]]:
    """``dict_clusters`` of the notebooks: cluster number (1, 2, ...) -> indices of its members.

    Works with the labels of ``fcluster`` (1..K) and of K-Means/K-Medoids (0..K-1).
    """
    clusters = np.asarray(clusters)
    return {k: np.flatnonzero(clusters == c).tolist() for k, c in enumerate(np.unique(clusters), 1)}


class KMeans:
    """``KMeans(n_clusters, random_state=...)`` as in scikit-learn 1.0/1.1 (Lloyd, n_init=10)."""

    def __init__(self, n_clusters: int = 8, random_state: int | None = None, n_init: int = 10,
                 max_iter: int = 300, tol: float = 1e-4):
        self.n_clusters, self.random_state = n_clusters, random_state
        self.n_init, self.max_iter, self.tol = n_init, max_iter, tol

    def _kmeans_plusplus(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """k-means++ seeding of scikit-learn <= 1.2 (X is centred)."""
        n_local_trials = 2 + int(np.log(self.n_clusters))
        sq_norms = np.einsum("ij,ij->i", X, X)

        def sq_distances(C):
            return np.maximum(np.einsum("ij,ij->i", C, C)[:, None] - 2 * C @ X.T + sq_norms[None, :], 0)

        indices = [rng.randint(X.shape[0])]
        closest = sq_distances(X[indices[0]][None, :])[0]
        potential = closest.sum()
        for _ in range(1, self.n_clusters):
            draws = rng.random_sample(n_local_trials) * potential
            candidates = np.searchsorted(np.cumsum(closest, dtype=np.float64), draws)
            np.clip(candidates, None, closest.size - 1, out=candidates)
            candidate_dist = np.minimum(closest, sq_distances(X[candidates]))
            candidate_pot = candidate_dist.sum(axis=1)
            best = np.argmin(candidate_pot)
            potential, closest = candidate_pot[best], candidate_dist[best]
            indices.append(candidates[best])
        return X[indices]

    def fit(self, X, y=None) -> "KMeans":
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)
        mean = X.mean(axis=0)
        best = None
        for _ in range(self.n_init):
            init = self._kmeans_plusplus(X - mean, rng) + mean
            model = _SklearnKMeans(self.n_clusters, init=init, n_init=1, max_iter=self.max_iter, tol=self.tol,
                                   algorithm="lloyd").fit(X)
            if best is None or model.inertia_ < best.inertia_:
                best = model
        self.labels_, self.cluster_centers_, self.inertia_ = best.labels_, best.cluster_centers_, best.inertia_
        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X).labels_


class KMedoids:
    """``KMedoids(n_clusters, metric, method="pam", init="k-medoids++", random_state)``."""

    def __init__(self, n_clusters: int = 8, metric: str = "euclidean", method: str = "pam",
                 init: str = "k-medoids++", max_iter: int = 300, random_state: int | None = None):
        if method != "pam" or init != "k-medoids++":
            raise NotImplementedError("only method='pam' with init='k-medoids++' is implemented")
        self.n_clusters, self.metric, self.method, self.init = n_clusters, metric, method, init
        self.max_iter, self.random_state = max_iter, random_state

    def _kpp_init(self, D: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        n_local_trials = 2 + int(np.log(self.n_clusters))
        medoids = np.empty(self.n_clusters, dtype=int)
        medoids[0] = rng.randint(D.shape[0])
        closest = D[medoids[0], :] ** 2
        potential = closest.sum()
        for c in range(1, self.n_clusters):
            draws = rng.random_sample(n_local_trials) * potential
            candidates = np.searchsorted(np.cumsum(closest, dtype=np.float64), draws)
            candidate_dist = D[candidates, :] ** 2
            best, best_pot, best_dist = None, np.inf, None
            for trial in range(n_local_trials):
                new_dist = np.minimum(closest, candidate_dist[trial])
                new_pot = new_dist.sum()
                if best is None or new_pot < best_pot:
                    best, best_pot, best_dist = candidates[trial], new_pot, new_dist
            medoids[c], potential, closest = best, best_pot, best_dist
        return medoids

    @staticmethod
    def _compute_optimal_swap(D, medoid_idxs, not_medoid_idxs, Djs, Ejs):
        """Swap (medoid, non-medoid) decreasing the total cost the most, or None.

        Vectorised version of the Cython function of scikit-learn-extra:
        candidates are scanned non-medoid first and the first strict minimum
        is kept.
        """
        D_hj = D[np.ix_(not_medoid_idxs, not_medoid_idxs)]
        Dj, Ej = Djs[not_medoid_idxs], Ejs[not_medoid_idxs]
        if_other = np.minimum(D_hj - Dj, 0.0)                 # j not attached to the swapped medoid
        if_attached = np.minimum(D_hj, Ej) - Dj - if_other    # extra cost if j is attached to it
        attached = (D[np.ix_(medoid_idxs, not_medoid_idxs)] == Dj).astype(np.float64)
        cost = if_other.sum(axis=1)[:, None] + if_attached @ attached.T  # [h, i]
        D_hi = D[np.ix_(not_medoid_idxs, medoid_idxs)]
        cost += np.where(D_hi < Ejs[medoid_idxs][None, :], D_hi, Ejs[medoid_idxs][None, :])
        h, i = divmod(int(np.argmin(cost)), len(medoid_idxs))
        return (medoid_idxs[i], not_medoid_idxs[h]) if cost[h, i] < 0 else None

    def fit(self, X, y=None) -> "KMedoids":
        X = np.asarray(X, dtype=np.float64)
        D = X if self.metric == "precomputed" else pairwise_distances(X, metric=self.metric)
        medoid_idxs = self._kpp_init(D, np.random.RandomState(self.random_state))
        Djs, Ejs = np.sort(D[medoid_idxs], axis=0)[[0, 1]]
        for self.n_iter_ in range(self.max_iter):
            not_medoid_idxs = np.delete(np.arange(len(D)), medoid_idxs)
            swap = self._compute_optimal_swap(D, medoid_idxs, not_medoid_idxs, Djs, Ejs)
            if swap is None:
                break
            medoid_idxs[medoid_idxs == swap[0]] = swap[1]
            Djs, Ejs = np.sort(D[medoid_idxs], axis=0)[[0, 1]]
        self.medoid_indices_ = medoid_idxs
        self.labels_ = np.argmin(D[medoid_idxs, :], axis=0)
        self.cluster_centers_ = None if self.metric == "precomputed" else X[medoid_idxs]
        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X).labels_
