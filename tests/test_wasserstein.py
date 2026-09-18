import numpy as np
import pytest

from tda_morphotypes.wasserstein import combine, decolored_dist, distance_matrix, wasserstein_distance

gw = pytest.importorskip("gudhi.wasserstein")
pytest.importorskip("ot")


def random_diagram(rng, n, essential=0):
    births = rng.uniform(0, 10, n)
    dgm = np.column_stack([births, births + rng.exponential(3, n)])
    if essential:
        dgm = np.vstack([dgm, np.column_stack([rng.uniform(0, 1, essential), np.full(essential, np.inf)])])
    return dgm


@pytest.mark.parametrize("order, internal_p", [(2.0, 2.0), (1.0, np.inf), (3.0, 1.0)])
def test_matches_gudhi(order, internal_p):
    rng = np.random.default_rng(0)
    for _ in range(50):
        X, Y = random_diagram(rng, rng.integers(0, 12)), random_diagram(rng, rng.integers(0, 12))
        expected = gw.wasserstein_distance(X, Y, order=order, internal_p=internal_p)
        assert wasserstein_distance(X, Y, order, internal_p) == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_essential_parts():
    rng = np.random.default_rng(1)
    for _ in range(20):
        X, Y = random_diagram(rng, 5, essential=1), random_diagram(rng, 3, essential=1)
        expected = gw.wasserstein_distance(X, Y, order=2, internal_p=2)
        assert wasserstein_distance(X, Y) == pytest.approx(expected, rel=1e-10)
    X, Y = random_diagram(rng, 3, essential=1), random_diagram(rng, 3, essential=2)
    assert wasserstein_distance(X, Y) == np.inf


def test_empty_and_symmetry():
    dgm = np.array([[0.0, 2.0], [1.0, 1.5]])
    empty = np.empty((0, 2))
    assert wasserstein_distance(empty, empty) == 0.0
    # each point goes to the diagonal: sum of ((d - b) / sqrt(2))^2
    assert wasserstein_distance(dgm, empty) == pytest.approx(np.sqrt((2.0 ** 2 + 0.5 ** 2) / 2))
    assert wasserstein_distance(dgm, empty) == wasserstein_distance(empty, dgm)


def test_decolored_dist():
    rng = np.random.default_rng(3)
    X1 = [random_diagram(rng, 2, essential=1), random_diagram(rng, 4), random_diagram(rng, 5)]
    X2 = [random_diagram(rng, 1, essential=1), random_diagram(rng, 3), random_diagram(rng, 6)]
    per_degree = decolored_dist(X1, X2, dim=[0, 1, 2])
    assert per_degree == pytest.approx([gw.wasserstein_distance(X1[d], X2[d], order=2, internal_p=2)
                                        for d in range(3)])
    assert decolored_dist(X1, X2) == pytest.approx(np.sqrt(np.sum(np.square(per_degree))))
    assert decolored_dist(X1, X2, dim=[2, 1]) == pytest.approx(per_degree[:0:-1])


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_distance_matrix_condensed_order(n_jobs):
    rng = np.random.default_rng(2)
    diagrams = [random_diagram(rng, rng.integers(1, 6)) for _ in range(7)]
    condensed = distance_matrix(diagrams, n_jobs=n_jobs)
    expected = [wasserstein_distance(diagrams[i], diagrams[j]) for i in range(7) for j in range(i + 1, 7)]
    np.testing.assert_allclose(condensed, expected)


def test_combine():
    per_degree = {0: np.array([3.0]), 1: np.array([4.0]), 2: np.array([12.0])}
    assert combine(per_degree, (0, 1)) == pytest.approx([5.0])
    assert combine(per_degree, (0, 1, 2)) == pytest.approx([13.0])
