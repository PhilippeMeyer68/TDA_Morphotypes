import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import davies_bouldin_score, silhouette_samples

from tda_morphotypes.anomalies import anomaly_percentages, best_range, solos
from tda_morphotypes.clustering import make_dict_clusters
from tda_morphotypes.indices import (GDI, DB_index, Dunn, cluster_diameter, cluster_distance_mean_medoid,
                                     cluster_mean_distance, cluster_table, local_optima, medoid, merge_heights,
                                     silhouette_index)


def test_gdi_extremes():
    # males are the first two individuals
    assert GDI({1: [0, 1], 2: [2, 3]}, male_nb_scans=2) == 1.0
    assert GDI({1: [0, 2], 2: [1, 3]}, male_nb_scans=2) == 0.0
    # clusters {m, m, f} and {f}: 2 * (|2/3 - 1/2| * 3 + |0 - 1/2| * 1) / 4
    assert GDI({1: [0, 1, 2], 2: [3]}, male_nb_scans=2) == pytest.approx(0.5)


def blobs(seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(c, 0.5, (12, 3)) for c in (0, 3, 9)]), np.repeat([1, 2, 3], 12)


def test_DB_index_matches_sklearn():
    sh, clusters = blobs()
    assert DB_index(make_dict_clusters(clusters), sh) == pytest.approx(davies_bouldin_score(sh, clusters))


def test_silhouette_index_and_Dunn():
    sh, clusters = blobs()
    D = squareform(pdist(sh))
    mixed = np.tile([1, 2, 3], 12)
    assert 0.5 < silhouette_index(make_dict_clusters(clusters), D) <= 1
    assert silhouette_index(make_dict_clusters(mixed), D) < 0.1
    # mean over clusters (singletons left out) of the per-point silhouettes
    with_singleton = np.r_[clusters[:-1], 4]
    samples = silhouette_samples(D, with_singleton, metric="precomputed")
    expected = np.mean([samples[with_singleton == c].mean() for c in (1, 2, 3)])
    assert silhouette_index(make_dict_clusters(with_singleton), D) == pytest.approx(expected)
    assert Dunn(make_dict_clusters(clusters), sh) > Dunn(make_dict_clusters(mixed), sh)


def test_cluster_descriptors():
    sh = np.array([[0.0], [1.0], [2.0], [10.0]])
    C = [0, 1, 2]
    assert medoid(C, sh) == 1
    assert cluster_diameter(C, sh) == 2
    assert cluster_mean_distance(C, sh) == pytest.approx(4 / 3)
    assert cluster_distance_mean_medoid(C, sh) == 0
    table = cluster_table({1: C, 2: [3]}, sh, ["a", "b", "c", "d"])
    assert table["size"].tolist() == [3, 1]
    assert table["medoid"].tolist() == ["b", "d"]


def test_local_optima():
    ks = [3, 4, 5, 6, 7, 8]
    assert local_optima(ks, [1.0, 2.0, 1.0, 3.0, 3.0, 2.0], maximize=True) == [4, 6, 7]
    assert local_optima(ks, [1.0, 2.0, 1.0, 3.0, 3.0, 2.0], maximize=False) == [5]


def test_merge_heights():
    Z = linkage(np.array([[0.0], [1.0], [10.0], [30.0]]), method="complete")
    np.testing.assert_allclose(merge_heights(Z, [2, 3, 4]), [30, 10, 1])


def test_anomalies_are_late_merging_solos():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (40, 2)), [[30, 30], [-30, 25]]])
    Z = linkage(pdist(X), method="complete")
    percentages = anomaly_percentages(Z, [40, 41])
    assert percentages["percent2"].iloc[-1] == 100
    start, end = best_range(percentages)
    assert percentages.set_index("nb_clus").loc[start, "percent3"] == 100
    assert set(solos(Z, start)) == {40, 41}
    assert end >= start
