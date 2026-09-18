import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import pairwise_distances

from tda_morphotypes.clustering import KMeans, KMedoids, make_dict_clusters


def pam_reference(D, n_clusters, random_state=42, max_iter=300):
    """KMedoids(method='pam', init='k-medoids++') of scikit-learn-extra 0.2.0, loop by loop."""
    rng = np.random.RandomState(random_state)
    medoids = KMedoids(n_clusters)._kpp_init(D, rng)
    Djs, Ejs = np.sort(D[medoids], axis=0)[[0, 1]]
    for _ in range(max_iter):
        others = np.delete(np.arange(len(D)), medoids)
        best = (1, 1, 0.0)
        for h in others:
            for i in medoids:
                cost = 0.0
                for j in others:
                    if D[i, j] == Djs[j]:
                        cost += (D[j, h] if D[h, j] < Ejs[j] else Ejs[j]) - Djs[j]
                    elif D[j, h] < Djs[j]:
                        cost += D[j, h] - Djs[j]
                cost += D[i, h] if D[h, i] < Ejs[i] else Ejs[i]
                if cost < best[2]:
                    best = (i, h, cost)
        if best[2] >= 0:
            break
        medoids[medoids == best[0]] = best[1]
        Djs, Ejs = np.sort(D[medoids], axis=0)[[0, 1]]
    return np.argmin(D[medoids, :], axis=0), medoids


def test_kmedoids_matches_reference_pam():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 1.0, (15, 3)) for c in (0, 4, 8)])
    D = pairwise_distances(X)
    for k in (2, 3, 5):
        kmed = KMedoids(n_clusters=k, method="pam", init="k-medoids++", random_state=42).fit(X)
        ref_labels, ref_medoids = pam_reference(D, k)
        assert list(kmed.medoid_indices_) == list(ref_medoids)
        assert list(kmed.labels_) == list(ref_labels)
        precomputed = KMedoids(n_clusters=k, metric="precomputed", random_state=42).fit(D)
        np.testing.assert_array_equal(precomputed.medoid_indices_, kmed.medoid_indices_)


def test_kmeans_finds_separated_groups_and_is_deterministic():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(c, 0.3, (20, 4)) for c in (0, 5, 10)])
    labels = KMeans(3, random_state=42).fit_predict(X)
    assert len(set(labels[:20])) == len(set(labels[20:40])) == len(set(labels[40:])) == 1
    assert len(set(labels)) == 3
    np.testing.assert_array_equal(labels, KMeans(3, random_state=42).fit_predict(X))


def test_make_dict_clusters():
    assert make_dict_clusters([1, 2, 1, 3]) == {1: [0, 2], 2: [1], 3: [3]}
    assert make_dict_clusters(np.array([0, 1, 0])) == {1: [0, 2], 2: [1]}


def test_ward_on_condensed_equals_ward_on_observations():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(c, 0.2, (10, 2)) for c in (0, 5)])
    condensed = pairwise_distances(X)[np.triu_indices(20, 1)]
    for method in ("complete", "ward"):
        clusters = fcluster(linkage(condensed, method=method), 2, criterion="maxclust")
        assert len(set(clusters[:10])) == len(set(clusters[10:])) == 1 and clusters[0] != clusters[-1]
    np.testing.assert_array_equal(fcluster(linkage(X, method="ward"), 2, criterion="maxclust"),
                                  fcluster(linkage(condensed, method="ward"), 2, criterion="maxclust"))
