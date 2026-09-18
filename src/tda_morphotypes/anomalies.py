"""Detection of scan anomalies as late-merging singletons (section 3).

When the complete-linkage dendrogram is cut at ``nb_clus`` clusters, the
individuals alone in their cluster (the *solos*) are compared with the known
anomalies:

* ``percent1``: percentage of solos that are anomalies (100 if there is no solo);
* ``percent2``: percentage of anomalies that are solos;
* ``percent3``: their mean, the truncation criterion of the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster


def solos(Z: np.ndarray, nb_clus: int) -> list[int]:
    """Indices of the individuals alone in their cluster at ``nb_clus`` clusters."""
    clusters = fcluster(Z, nb_clus, criterion="maxclust")
    counts = np.bincount(clusters)
    return np.flatnonzero(counts[clusters] == 1).tolist()


def anomaly_percentages(Z: np.ndarray, anomalies) -> pd.DataFrame:
    """percent1, percent2 and percent3 for nb_clus = 1, 2, ... until every anomaly is a solo."""
    rows = []
    nb_clus = 1
    while not rows or rows[-1]["percent2"] < 100:
        isolated = solos(Z, nb_clus)
        nb_anomalies = sum(i in anomalies for i in isolated)
        percent1 = 100 * nb_anomalies / len(isolated) if isolated else 100.0
        percent2 = 100 * nb_anomalies / len(anomalies)
        rows.append({"nb_clus": nb_clus, "nb_solos": len(isolated), "nb_anomalies": nb_anomalies,
                     "percent1": percent1, "percent2": percent2, "percent3": (percent1 + percent2) / 2})
        nb_clus += 1
    return pd.DataFrame(rows)


def best_range(percentages: pd.DataFrame) -> tuple[int, int]:
    """Range of numbers of clusters where percent3 reaches its maximum (first plateau)."""
    percent3 = percentages["percent3"].tolist()
    j = int(np.argmax(percent3))
    i = j
    while i < len(percent3) and percent3[i] == percent3[j]:
        i += 1
    nb_clus = percentages["nb_clus"].tolist()
    return nb_clus[j], nb_clus[i - 1]
