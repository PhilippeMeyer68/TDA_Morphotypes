import numpy as np
import pytest
from gudhi.representations import Silhouette as GudhiSilhouette

from tda_morphotypes.config import SilhouetteSetting
from tda_morphotypes.persistence import DiagramSet
from tda_morphotypes.silhouettes import WEIGHTS, Silhouette, silhouette_vectors


def gudhi36_silhouettes(diagrams, resolution, weight):
    """Silhouette(resolution, weight).fit_transform of gudhi 3.6 (used for the paper), transcribed."""
    P = np.concatenate(diagrams)
    lo, hi = P[:, 0].min(), P[:, 1].max()
    x = np.linspace(lo, hi, resolution)
    step = x[1] - x[0]
    out = []
    for D in diagrams:
        sh = np.zeros(resolution)
        w = np.array([weight(p) for p in D])
        for (px, py), wj in zip(D, w / w.sum()):
            mn = int(np.clip(np.ceil((px - lo) / step), 0, resolution))
            md = int(np.clip(np.ceil((0.5 * (py + px) - lo) / step), 0, resolution))
            mx = int(np.clip(np.ceil((py - lo) / step), 0, resolution))
            if mn < resolution and mx > 0:
                v = lo + mn * step - px
                for k in range(mn, md):
                    sh[k] += wj * v
                    v += step
                v = py - lo - md * step
                for k in range(md, mx):
                    sh[k] += wj * v
                    v -= step
        out.append(np.sqrt(2) * sh)
    return np.array(out)


def random_diagrams(seed, n=6):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        b = rng.uniform(1, 10, rng.integers(1, 8))
        out.append(np.column_stack([b, b + rng.exponential(4, len(b))]))
    return out


@pytest.mark.parametrize("weight", ["uniform", "birth", "ratio"])
def test_matches_gudhi_36(weight):
    diagrams = random_diagrams(0)
    expected = gudhi36_silhouettes(diagrams, 250, WEIGHTS[weight])
    result = Silhouette(resolution=250, weight=weight).fit_transform(diagrams)
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-12)


def test_matches_recent_gudhi_with_endpoints():
    diagrams = random_diagrams(1)
    expected = GudhiSilhouette(resolution=100, weight=lambda x: x[0], keep_endpoints=True).fit_transform(diagrams)
    result = Silhouette(resolution=100, weight=lambda x: x[0]).fit_transform(diagrams)
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-12)


def test_empty_diagrams():
    diagrams = random_diagrams(2) + [np.empty((0, 2))]
    result = Silhouette(resolution=50).fit_transform(diagrams)
    assert np.all(result[-1] == 0)
    assert np.all(Silhouette(resolution=10).fit_transform([np.empty((0, 2))] * 3) == 0)


def test_vectors_layout_and_standardisation():
    rng = np.random.default_rng(3)
    names, diagrams = [], []
    for i in range(5):
        h0 = np.vstack([np.column_stack([np.zeros(2), rng.uniform(1, 2, 2)]), [[0.0, np.inf]]])
        h1, h2 = random_diagrams(10 + i, 2)
        names.append(f"S{i}")
        diagrams.append([h0, h1, h2])
    X_decolor = DiagramSet(names, diagrams)
    setting = SilhouetteSetting(resolutions=(5, 7, 9), zero_degrees=(1,))
    raw = silhouette_vectors(X_decolor, setting, standardize=False)
    assert raw.shape == (5, 21)
    assert np.all(raw[:, 5:12] == 0)
    test0 = [xd[0][:-1] for xd in diagrams]  # the essential class is the last H0 interval
    np.testing.assert_allclose(raw[:, :5], Silhouette(resolution=5).fit_transform(test0))
    np.testing.assert_array_equal(silhouette_vectors(diagrams, setting, standardize=False), raw)
    sh = silhouette_vectors(X_decolor, setting)
    assert sh.mean() == pytest.approx(0) and sh.std() == pytest.approx(1)
