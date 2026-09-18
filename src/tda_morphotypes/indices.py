"""Clustering quality indices and cluster descriptors (sections 4 and 5).

The functions follow the notebooks of the paper; a clustering is given as
``dict_clusters`` (cluster number -> list of indices, see
``clustering.make_dict_clusters``) and the individuals by their vectors ``sh``
(silhouettes) or by a square distance matrix ``D``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def GDI(dict_clusters, male_nb_scans: int) -> float:
    """Gender discrimination index (equation 4); males are the indices below ``male_nb_scans``."""
    nb_scans = sum(len(C) for C in dict_clusters.values())
    test = 0.0
    for C in dict_clusters.values():
        male_percentage = np.sum(np.array(C) < male_nb_scans) / len(C)
        test += 2 * abs(male_percentage - 0.5) * len(C)
    return test / nb_scans


def cluster_centre(C, sh) -> np.ndarray:
    """Barycentre of the members of a cluster."""
    return sh[C].mean(axis=0)


def cluster_distance_centre(C, sh) -> float:
    """Mean distance between the members of a cluster and its barycentre."""
    return float(np.linalg.norm(sh[C] - cluster_centre(C, sh), axis=1).mean())


def DB_index(dict_clusters, sh) -> float:
    """Davies-Bouldin index (lower is better)."""
    clusters = list(dict_clusters.values())
    centres = np.array([cluster_centre(C, sh) for C in clusters])
    spread = np.array([cluster_distance_centre(C, sh) for C in clusters])
    separation = squareform(pdist(centres))
    ratios = (spread[:, None] + spread[None, :]) / np.where(separation > 0, separation, np.inf)
    np.fill_diagonal(ratios, -np.inf)
    return float(ratios.max(axis=1).mean())


def silhouette_index(dict_clusters, D) -> float:
    """Mean over clusters of the mean silhouette of their points (higher is better).

    ``D`` is the square distance matrix. Singleton clusters are left out of the
    average, as in the original notebooks (this differs from
    ``sklearn.metrics.silhouette_score``, which averages over points).
    """
    clusters = list(dict_clusters.values())
    silh_list = []
    for k, C in enumerate(clusters):
        if len(C) < 2:
            continue
        a = D[np.ix_(C, C)].sum(axis=1) / (len(C) - 1)       # mean distance to its own cluster
        b = np.min([D[np.ix_(C, C2)].mean(axis=1) for k2, C2 in enumerate(clusters) if k2 != k], axis=0)
        silh_list.append(np.mean((b - a) / np.maximum(a, b)))
    return float(np.mean(silh_list))


def cluster_diameter(C, sh) -> float:
    """Largest distance between two members of a cluster."""
    return float(pdist(sh[C]).max()) if len(C) > 1 else 0.0


def Dunn(dict_clusters, sh) -> float:
    """Smallest distance between barycentres over largest diameter (higher is better)."""
    clusters = list(dict_clusters.values())
    centres = np.array([cluster_centre(C, sh) for C in clusters])
    return float(pdist(centres).min() / max(cluster_diameter(C, sh) for C in clusters))


def medoid(C, sh) -> int:
    """Member of a cluster minimising the sum of the distances to the others."""
    if len(C) == 1:
        return C[0]
    return C[int(np.argmin(squareform(pdist(sh[C])).sum(axis=0)))]


def cluster_mean_distance(C, sh) -> float:
    """Mean distance between two members of a cluster."""
    return float(pdist(sh[C]).mean()) if len(C) > 1 else 0.0


def cluster_distance_mean_medoid(C, sh) -> float:
    """Distance between the barycentre and the medoid of a cluster."""
    return float(np.linalg.norm(sh[medoid(C, sh)] - cluster_centre(C, sh)))


def merge_heights(Z: np.ndarray, n_clusters) -> np.ndarray:
    """Height of the merge leading from k to k-1 clusters, for k in ``n_clusters`` (elbow method)."""
    heights = Z[::-1, 2]  # heights[0] is the last merge (2 -> 1 cluster)
    return np.array([heights[k - 2] for k in n_clusters])


def local_optima(n_clusters, values, maximize: bool) -> list[int]:
    """Numbers of clusters where an index reaches a local optimum (plateau ends included)."""
    v = np.asarray(values) if maximize else -np.asarray(values)
    return [int(n_clusters[i]) for i in range(1, len(v) - 1)
            if v[i] >= v[i - 1] and v[i] >= v[i + 1] and (v[i] > v[i - 1] or v[i] > v[i + 1])]


def cluster_table(dict_clusters, sh, names) -> pd.DataFrame:
    """Descriptors of Tables 3 and 4, one row per cluster."""
    nb_scans = sum(len(C) for C in dict_clusters.values())
    return pd.DataFrame([{
        "cluster": f"C{key}",
        "size": len(C),
        "proportion_pct": 100 * len(C) / nb_scans,
        "mean_distance": cluster_mean_distance(C, sh),
        "diameter": cluster_diameter(C, sh),
        "distance_to_mean": cluster_distance_centre(C, sh),
        "distance_mean_medoid": cluster_distance_mean_medoid(C, sh),
        "medoid": names[medoid(C, sh)],
    } for key, C in dict_clusters.items()])
