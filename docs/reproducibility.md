# Reproducibility notes

This repository is a rewrite of the code behind

> S. de Rose, P. Meyer, F. Bertrand. *Human Body Shapes Anomaly Detection and
> Classification Using Persistent Homology.* Algorithms 16(3), 161, 2023.
> <https://doi.org/10.3390/a16030161>

The published numbers were produced in 2022 by two sets of Jupyter notebooks:
the internship repository of S. de Rose (section 4 and the illustrations of
section 2; its last version is in the git history of this repository, commit
`15459c7`) and the notebooks of P. Meyer (sections 3 and 5). The two sets
did not preprocess the scans in exactly the same way, and a few results depend
on the behaviour of the library versions of the time. This page lists what was
recovered from the original code and its saved outputs, and how each value of
the paper is reproduced.

## Data

The 1517 male and 1531 female meshes of the SPRING dataset (Yang et al., 2014),
derived from CAESAR: 12,500 vertices and 25,000 triangles each, in metres, the
vertical axis being z. Scans are processed in lexicographic order of their file
names, males first when both sexes are combined.

## Persistence diagrams

Alpha complexes (gudhi), homology in degrees 0, 1 and 2. Filtration values are
squared radii. Three preprocessings are used:

| Setting | Used for | Normalisation | Minimum persistence |
|---|---|---|---|
| `body_m` | sections 3 and 5 | x 1.7 / h, h = height **rounded to 2 decimals** (m) | 3e-4 m^2 |
| `body_cm` | section 4, diagrams | x 170 / h (h not rounded) | 3 cm^2 |
| `body_cm` filtered | section 4, body silhouettes | same | 5.25 cm^2 |
| `trunk` | section 4.2 | trunk isolation, x 170 / h | 5500 / 999 = 5.5055 cm^2 |

The two body normalisations are the same homothety up to the rounding of the
height, which changes a few features near the threshold; each part of the
paper is reproduced with its own setting. A feature is kept when its
persistence is strictly larger than the threshold (gudhi's convention).

The trunk isolation (`geometry.body_to_trunk`) is the original algorithm with
its constants. It keeps the 10,990 lowest vertices, restricts them to an 80 cm
band below the highest of them (the [66.5, 146.5] cm range quoted in the paper
for a typical individual) and cuts the arms with two pairs of lines of the
(x, z) plane.

## Distances

The (2, 2)-Wasserstein distance is computed per degree; the distance between
two scans is sqrt(W0^2 + W1^2 + W2^2).

* **Section 3 (anomalies).** In the original distance matrices the H0 term is
  zero for every pair of scans. This is an artefact of the gudhi/POT versions of
  the time, triggered by the essential class (0, inf) that every H0 diagram
  contains. The anomaly detection of the paper therefore uses H1 and H2 only
  (`config.ANOMALY_DEGREES`). The H0 term is small: all H0 features are born at
  0 and die before 1.1e-3 m^2.
* **Section 4.** All three degrees (`config.GDI_DEGREES`).

`wasserstein.wasserstein_distance` solves the optimal partial matching exactly
as an assignment problem. It agrees with `gudhi.wasserstein` to machine
precision (see `tests/test_wasserstein.py`) and is several times faster.

## Silhouettes (sections 2.2, 4 and 5)

For each degree, the silhouette is sampled at 25 (H0) and 250 (H1, H2) points.
The essential H0 class is left out. The 525 values are concatenated and
standardised by the global mean and standard deviation of the matrix.

* The sampling range is [smallest birth, largest death] over the whole set of
  diagrams, as in gudhi <= 3.7. Recent versions shrink it by half a step unless
  `keep_endpoints=True`, and return NaN for empty diagrams. The silhouettes are
  therefore implemented in `silhouettes.Silhouette`, which keeps the gudhi
  interface and is tested against the gudhi 3.6 algorithm.
* Weights:
  * sections 2.2 and 5: 1 for H0 and the birth time for H1 and H2;
  * trunk silhouettes of section 4.2: (d + b) / (d - b) for every degree.
* **Body silhouettes of section 4 (Figure 14, Table 2).** In the original run
  every H1 silhouette was zero. Some empty H1 diagrams had been saved as 1-D
  arrays, which made gudhi's automatic sampling range fail silently (gudhi
  issue #507). This is reproduced with `GDI_BODY_SILHOUETTE.zero_degrees = (1,)`.
  With the H1 block restored, the Ward curve changes by at most 0.022 and its
  mean goes from 0.7378 to 0.7377.

## Clustering

* Hierarchical clusterings: `scipy.cluster.hierarchy.linkage`. Ward's method on
  Wasserstein distances goes through the Lance-Williams update. Trees are cut
  with `fcluster(..., criterion="maxclust")`.
* K-Means (Figures 14 and 17): scikit-learn 1.0/1.1 behaviour with
  `random_state=42`. Ten k-means++ initialisations are drawn one after the
  other from a single `RandomState(42)`, the first centre with `randint`;
  scikit-learn >= 1.3 draws it differently. `clustering.KMeans` rebuilds these
  initialisations and runs scikit-learn's Lloyd iterations from them.
* K-Medoids: `KMedoids(method="pam", init="k-medoids++", random_state=42)` of
  scikit-learn-extra 0.2.0, which no longer builds on current Python. The swap
  step is vectorised in `clustering.KMedoids`; `tests/test_clustering.py`
  checks it against a line-by-line transcription of the original Cython loop.

## Indices of section 5

* Davies-Bouldin: Euclidean centroids, identical to scikit-learn's.
* Silhouette index: the mean over clusters of the mean silhouette of their
  points; singleton clusters are left out.
* Dunn: smallest distance between centroids over largest cluster diameter.

The number of clusters is 8 for men and 7 for women, as in the paper.

## Known differences with the paper

* Table 4, cluster C3 (403 women): the paper prints 27 %. The original notebook
  divided every female cluster size by 1513 instead of 1527 (1531 scans minus 4
  anomalies); 403 / 1527 = 26.4 %. This is the only difference found.
* The paper numbers homology classes in two ways. Figures 2 and 5 follow the
  barcode order (decreasing birth, H0 last). Figures 9 and 10 number the
  classes within each degree by increasing birth: H2 n°3, 5 and 6 of S2962 are
  its two legs and its principal H2 class. `representatives.HomologyClass`
  exposes both numberings.
* The meshes are rendered with a simple orthographic front view with the
  heads removed. The pictures are not pixel-identical to those of the paper.

## Verification

The rewrite was checked against the outputs saved in 2022: the pickled
diagrams, distances and GDI curves of the internship repository (git history)
and of P. Meyer's notebooks.

| Quantity | Comparison | Result |
|---|---|---|
| Diagrams (4 settings x 3048 scans) | number of points per degree | identical for every scan |
| | values | relative difference <= 2e-11 |
| Anomaly distances (H1, H2), 2 x ~1.15 M pairs | values | relative difference <= 1e-10 |
| Trunk distances (H0, H1, H2), 3 x 4.6 M pairs | values | relative difference <= 1e-10 |
| GDI curves, 12 curves x 29 values | values | identical (difference <= 1e-12) |
| Tables 1-4, anomaly ranges, detected anomalies, medoids | as printed in the paper | identical except Table 4 C3 (see above) |

The residual differences on the diagrams come from the CGAL/gudhi versions
used to compute the alpha filtration values. They change none of the results.

`tda-morphotypes report` repeats the comparison with the paper and the GDI
curves (`results/comparison_with_paper.csv`).
