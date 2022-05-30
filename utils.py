import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import gudhi
import gudhi.wasserstein
import meshio
import os
import networkx as nx


class Graph(nx.Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def add_face(self, face):
        self.add_edge(face[0], face[1])
        self.add_edge(face[0], face[2])
        self.add_edge(face[1], face[2])

    def add_tetra(self, tetra):
        edges = []
        for i in range(4):
            for j in range(i+1, 4):
                edges.append([tetra[i], tetra[j]])
        self.add_edges_from(edges)


class point_set:
    def __init__(self, points: np.ndarray) -> None:
        self.points = points.copy()
        self.nb, self.dim = self.points.shape
        self.__alpha_complex = None
        self.__simplex_tree = None
        self.__persistence = None
        self.__decolored = None
        self.__min_persistence = 0

    def to_plot(self):
        return self.points.T

    def plot(self, size=2., color='blue', opacity=1., title=None, width=1000, height=1000):
        to_plot = self.to_plot()
        if self.dim == 3:
            fig = go.Figure(data=go.Scatter3d(
                x=-to_plot[0],
                y=-to_plot[1],
                z=to_plot[2],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    opacity=opacity
                )
            ))

            fig.update_layout(width=width, height=height, title=title)
            fig.show()
        else:
            fig = plt.figure(figsize=(8, 8))

            plt.scatter(to_plot[0], to_plot[1], s=size, c=color, alpha=opacity)
            plt.title(title)
            plt.show()

    def AlphaComplex(self):
        if self.__alpha_complex == None:
            self.__alpha_complex = gudhi.AlphaComplex(points=self.points)
        return self.__alpha_complex

    def SimplexTree(self):
        if self.__simplex_tree == None:
            self.__simplex_tree = self.AlphaComplex().create_simplex_tree()
        return self.__simplex_tree

    def Persistence(self, min_persistence=None):
        if self.__persistence == None:
            if min_persistence == None:
                min_persistence = 0
            self.__persistence = self.SimplexTree().persistence(
                min_persistence=min_persistence)
            self.__min_persistence = min_persistence
        elif min_persistence != None and self.__min_persistence != min_persistence:
            self.__persistence = self.SimplexTree().persistence(
                min_persistence=min_persistence)
            self.__min_persistence = min_persistence
        return self.__persistence

    def DecoloredPersistence(self, min_persistence=None):
        if self.__min_persistence == None or min_persistence != self.__min_persistence:
            self.Persistence(min_persistence=min_persistence)
            self.__decolored = [
                np.array(
                    self.SimplexTree().persistence_intervals_in_dimension(0).tolist()),
                np.array(
                    self.SimplexTree().persistence_intervals_in_dimension(1).tolist()),
                np.array(
                    self.SimplexTree().persistence_intervals_in_dimension(2).tolist())
            ]
        return self.__decolored

    def PlotPersistenceDiagram(self, axes=None):
        if axes == None:
            _, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
        gudhi.plot_persistence_diagram(
            persistence=self.Persistence(), legend=True, axes=axes)

    def PlotPersistenceBarcode(self, axes=None):
        if axes == None:
            _, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
        gudhi.plot_persistence_barcode(
            persistence=self.Persistence(), axes=axes)

    def height(self):
        min_ = np.inf
        max_ = 0

        for row in self.points:
            if row[2] < min_:
                min_ = row[2]
            if row[2] > max_:
                max_ = row[2]

        return max_ - min_

    def normalize(self, size=170):
        coeff = size / self.height()
        self.points *= coeff


def decolored_dist(X1, X2, dim=None, p=2., order=2.):
    if dim == None:
        dist1 = 0
        if len(X1[0]) > 1 or len(X2[0]) > 1:
            dist1 = gudhi.wasserstein.wasserstein_distance(
                X1[0], X2[0], order=order, internal_p=p)
        dist2 = 0
        if len(X1[1]) > 0 or len(X2[1]) > 0:
            dist2 = gudhi.wasserstein.wasserstein_distance(
                X1[1], X2[1], order=order, internal_p=p)
        dist3 = 0
        if len(X1[2]) > 0 or len(X2[2]) > 0:
            dist3 = gudhi.wasserstein.wasserstein_distance(
                X1[2], X2[2], order=order, internal_p=p)

        temp = np.power(dist1, p) + np.power(dist2, p) + np.power(dist2, p)
        return np.power(temp, 1/p)

    result = np.zeros_like(dim, dtype=float)
    if 0 in dim:
        i = dim.index(0)
        result[i] = 0
        if len(X1[0]) > 1 or len(X2[0]) > 1:
            result[i] = gudhi.wasserstein.wasserstein_distance(
                X1[0], X2[0], order=p, internal_p=p)
    if 1 in dim:
        i = dim.index(1)
        result[i] = 0
        if len(X1[1]) > 0 or len(X2[1]) > 0:
            result[i] = gudhi.wasserstein.wasserstein_distance(
                X1[1], X2[1], order=p, internal_p=p)
    if 2 in dim:
        i = dim.index(2)
        result[i] = 0
        if len(X1[2]) > 0 or len(X2[2]) > 0:
            result[i] = gudhi.wasserstein.wasserstein_distance(
                X1[2], X2[2], order=p, internal_p=p)

    return result.tolist()


def tup_sort(a, b):
    if a < b:
        return (a, b)
    return (b, a)


def cluster_centre(C, dict_dist, p=2):
    min_centre = None
    min_distance = np.inf

    for i in C:
        dist = 0
        for j in C:
            if j != i:
                dist += np.power(dict_dist[tup_sort(i, j)], p)
        if dist < min_distance:
            min_distance = dist
            min_centre = i

    return min_centre


def cluster_medoid(C, dict_dist):
    min_centre = None
    min_distance = np.inf

    for i in C:
        dist = 0
        for j in C:
            if j != i:
                dist += dict_dist[tup_sort(i, j)]
        if dist < min_distance:
            min_distance = dist
            min_centre = i

    return min_centre


def dist_moyenne_cluster_centre(C, dict_dist):
    centre = cluster_centre(C, dict_dist)
    temp = 0
    for i in C:
        if i != centre:
            temp += dict_dist[tup_sort(i, centre)]

    return temp / len(C)


def indice_DB(dict_clusters, dict_dist):
    SDB = 0
    for C1 in dict_clusters:
        i = dict_clusters[C1]
        temp1 = dist_moyenne_cluster_centre(i, dict_dist)
        val = -np.inf
        for C2 in dict_clusters:
            j = dict_clusters[C2]
            if C2 != C1:
                temp2 = dist_moyenne_cluster_centre(j, dict_dist)
                temp3 = dict_dist[tup_sort(cluster_centre(
                    i, dict_dist), cluster_centre(j, dict_dist))]
                temp4 = (temp1 + temp2) / temp3
                if temp4 > val:
                    val = temp4
        SDB += val
    SDB = SDB / len(dict_clusters)

    return(SDB)


if __name__ == '__main__':
    pathglob = 'E:\Projet-M2\\2021-m2-ditex-morphotypes\Data\Male\\'  # À CHANGER
    # pathglob = '/mnt/e/Projet-M2/2021-m2-ditex-morphotypes/Data/Male/'  # À CHANGER
    all_scans = os.listdir(pathglob)

    def scan_from_index(i):
        path = pathglob + str(all_scans[i])
        scan = meshio.read(path)
        scan = scan.points
        return scan

    # spl = np.random.choice(range(len(all_scans)), 2, replace=False)
    spl = [1487, 1057]

    min_persistence = 3
    radius = np.sqrt(min_persistence)
    scan = scan_from_index(spl[0])
    PS1 = point_set(scan)
    PS1.normalize(PS1.height()*100)

    decolored_PS1 = PS1.DecoloredPersistence(min_persistence=min_persistence)

    scan = scan_from_index(spl[1])
    PS2 = point_set(scan)
    PS2.normalize(PS2.height()*100)

    decolored_PS2 = PS2.DecoloredPersistence(min_persistence=min_persistence)

    print("\nSCANS "+str(spl[0])+" ET "+str(spl[1]))
    a, b, c = decolored_dist(decolored_PS1, decolored_PS2, [0, 1, 2])
    print("Distance entre diagrammes :", np.sqrt(a**2+b**2+c**2))

    # fig = plt.figure(figsize=(12, 5))
    # fig.suptitle(spl[0])
    # PS1.PlotPersistenceDiagram(axes=fig.add_subplot(121))
    # PS1.PlotPersistenceBarcode(axes=fig.add_subplot(122))
    # plt.show()

    # fig = plt.figure(figsize=(12, 5))
    # fig.suptitle(spl[1])
    # PS2.PlotPersistenceDiagram(axes=fig.add_subplot(121))
    # PS2.PlotPersistenceBarcode(axes=fig.add_subplot(122))
    # plt.show()
