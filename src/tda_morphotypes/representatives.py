"""Geometric representatives of homology classes (section 2.3, Figures 5 and 10).

Port of ``find_homologies`` of the original ``Homology_Plot`` notebook, based
on gudhi's persistence pairs.

For each point of the persistence diagram of an alpha complex:

* H0: the vertex whose connected component dies;
* H1: the loop formed by the edge creating the class and a shortest path
  joining its endpoints among the edges already present (edges are weighted
  by their radius), together with the triangle killing the class;
* H2: the triangle creating the class and the tetrahedron killing it.

The paper uses two numberings: ``number``, as on the barcode of Figure 2
(gudhi's barcode plot: decreasing birth time, the H0 classes coming last),
and ``degree_number``, as in Figures 9 and 10 (1, 2, ... by increasing birth
time within each degree).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .persistence import point_set


@dataclass
class HomologyClass:
    number: int
    degree: int
    birth: float
    death: float
    birth_simplex: np.ndarray            # coordinates, shape (degree + 1, 3)
    death_simplex: np.ndarray | None     # coordinates, shape (degree + 2, 3); None if essential
    loop: np.ndarray | None = None       # H1 only: closed polygon, shape (m, 3)
    degree_number: int = -1


def _shortest_path(edges: np.ndarray, weights: np.ndarray, n_vertices: int, source: int, target: int) -> list[int]:
    graph = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n_vertices, n_vertices)).tocsr()
    _, predecessors = dijkstra(graph, directed=False, indices=source, return_predecessors=True)
    path = [target]
    while path[-1] != source:
        previous = predecessors[path[-1]]
        if previous < 0:
            raise RuntimeError("the endpoints of an H1 birth edge are not connected")
        path.append(previous)
    return path[::-1]


def find_homologies(PS: point_set, dimension: int | None = None) -> list[HomologyClass]:
    """Homology classes of ``PS`` with their representatives, all of them or those of one degree.

    As in the original notebook, ``PS.Persistence(min_persistence)`` must have
    been called first.
    """
    tree = PS.SimplexTree()
    PS.Persistence()
    alpha = PS.AlphaComplex()
    coords = np.array([alpha.get_point(v) for v in range(tree.num_vertices())])

    edges, radii = [], []
    for simplex, value in tree.get_skeleton(1):
        if len(simplex) == 2:
            edges.append(simplex)
            radii.append(value)
    edges, radii = np.array(edges), np.array(radii)

    classes = []
    for birth_simplex, death_simplex in tree.persistence_pairs():
        birth = tree.filtration(birth_simplex)
        death = tree.filtration(death_simplex) if death_simplex else np.inf
        degree = len(birth_simplex) - 1
        loop = None
        if degree == 1:
            before = radii < birth
            path = _shortest_path(edges[before], np.sqrt(radii[before]), len(coords), *birth_simplex)
            loop = coords[path + [path[0]]]
        classes.append(HomologyClass(
            number=-1, degree=degree, birth=birth, death=death,
            birth_simplex=coords[birth_simplex],
            death_simplex=coords[death_simplex] if death_simplex else None,
            loop=loop,
        ))
    classes.sort(key=lambda c: c.birth)
    counters = {}
    for rank, c in enumerate(classes):
        c.number = len(classes) - 1 - rank
        counters[c.degree] = c.degree_number = counters.get(c.degree, 0) + 1
    classes.sort(key=lambda c: c.number)
    return [c for c in classes if dimension is None or c.degree == dimension]
