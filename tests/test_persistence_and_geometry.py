import numpy as np
import pytest

from tda_morphotypes.geometry import normalize_scan, scan_height
from tda_morphotypes.persistence import DiagramSet, decolored_diag, point_set
from tda_morphotypes.representatives import find_homologies


def circle(n=60, radius=1.0, seed=0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(seed)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), 1e-3 * rng.standard_normal(n)])


def sphere(n=400, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, 3))
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def test_decolored_diagrams_of_circle_and_sphere():
    h0, h1, h2 = decolored_diag(circle(), min_persistence=0.1)
    assert len(h1) == 1 and len(h2) == 0
    assert np.isinf(h0[:, 1]).sum() == 1 and np.isinf(h0[-1, 1])  # the essential class comes last
    assert h1[0, 1] == pytest.approx(1.0, rel=1e-2)  # squared radius of the circle
    _, _, h2 = decolored_diag(sphere(), min_persistence=0.1)
    assert len(h2) == 1


def test_point_set():
    PS = point_set(circle() * 3)
    PS.normalize(size=1.0)
    assert PS.height() == pytest.approx(1.0)
    diagram = PS.Persistence(min_persistence=1e-3)
    assert sum(1 for d, _ in diagram if d == 1) == 1
    for a, b in zip(PS.DecoloredPersistence(), decolored_diag(PS.points, 1e-3)):
        np.testing.assert_array_equal(a, b)


def test_diagram_set_roundtrip_and_filter(tmp_path):
    diagrams = [decolored_diag(circle(seed=s), 0.0) for s in range(3)]
    ds = DiagramSet(["a", "b", "c"], diagrams)
    ds.save(tmp_path / "x.npz")
    loaded = DiagramSet.load(tmp_path / "x.npz")
    assert loaded.names == ds.names
    for d1, d2 in zip(loaded.diagrams, ds.diagrams):
        for a, b in zip(d1, d2):
            np.testing.assert_array_equal(a, b)
    filtered = ds.filtered(0.1)
    for d, direct in zip(filtered.diagrams, (decolored_diag(circle(seed=s), 0.1) for s in range(3))):
        for a, b in zip(d, direct):
            np.testing.assert_array_equal(np.sort(a, axis=0), np.sort(b, axis=0))
    assert ds.without(["b"]).names == ["a", "c"]
    assert (ds + ds.subset([0])).names == ["a", "b", "c", "a"]


def test_normalize_scan():
    scan = circle() + [0, 0, 5]
    scan[0, 2] = 7.3456
    assert scan_height(normalize_scan(scan, 170.0)) == pytest.approx(170.0)
    rounded = normalize_scan(scan, 1.7, decimals=2)
    assert scan_height(rounded) == pytest.approx(1.7 * scan_height(scan) / np.round(scan_height(scan), 2))


def test_find_homologies_of_circle():
    PS = point_set(circle())
    PS.Persistence(min_persistence=0.1)
    homologies = find_homologies(PS)
    loop = find_homologies(PS, dimension=1)[0]
    assert np.allclose(loop.loop[0], loop.loop[-1])
    assert len(loop.loop) > 10  # goes round the circle
    assert loop.death_simplex.shape == (3, 3)
    assert homologies[-1].degree == 0 and np.isinf(homologies[-1].death)
    assert [c.number for c in homologies] == list(range(len(homologies)))
