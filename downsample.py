import os
from unittest import result


def importer(module, *args):
    def importer_(module):
        from importlib import import_module
        from subprocess import check_call
        from sys import executable

        if '.' in module:
            index = module.index('.')
            package = module[:index]
            module = module[index:]
        else:
            package = None
        try:
            return import_module(module, package)
        except:
            check_call([executable, '-m', 'pip', 'install', '-U', package])
        finally:
            return import_module(module, package)
    if len(args) == 0:
        return importer_(module)    
    result = [getattr(importer(module), fun) for fun in args]
    if len(args) == 1:
        return result[0]
    return tuple(result)


np = importer('numpy')
plt = importer('matplotlib.pyplot')
go = importer('plotly.graph_objects')
gudhi = importer('gudhi')
meshio = importer('meshio')
gudhi.wasserstein = importer('gudhi.wasserstein')


class points_set:
    def __init__(self, points: np.ndarray) -> None:
        self.points = points.copy()
        self.nb, self.dim = self.points.shape
        self.tau = 1
        self.__alpha_complex = None
        self.__simplex_tree = None
        self.__persistence = None
        self.__decolored = None
        self.__min_persistence = 0

    def __boule(self, radius: float, N: np.ndarray, center: np.ndarray, out=True) -> np.ndarray:
        sq_rad = radius * radius
        if out:
            temp = np.where(np.sum((N - center)**2, axis=1) >= sq_rad)[0]
        else:
            temp = np.where(np.sum((N - center)**2, axis=1) < sq_rad)[0]
        return np.array([N[index] for index in temp])

    def __comp_gamma(self, t: int, radius: float):
        return radius**3 * np.exp(-t / self.tau)

    def __m_i(self, N_i: np.ndarray, s_i: np.ndarray, gamma: float) -> np.ndarray:
        m_i = np.zeros_like(s_i)
        for s_j in N_i:
            temp = s_j - s_i
            norm = np.linalg.norm(temp)
            m_i -= temp * gamma / norm**3

        return m_i

    def __hat_s_i(self, s_i: np.ndarray, m_i: float) -> np.ndarray:
        sm_i = s_i + m_i
        temp = np.sum((self.points - sm_i)**2, axis=1)

        return self.points[np.argmin(temp)]

    def downsample(self, npts: int, radius=np.sqrt(3)) -> np.ndarray:
        n = self.nb
        gamma = 1
        rtenth = radius * 0.1
        t = 1
        N = self.points.copy()
        N_r = np.array([N for i in np.random.choice(range(n), npts, replace=False)])
        dmat = np.zeros((npts, npts))

        for i in range(npts):
            print(i)
            for j in range(i+1, npts):
                dmat[i, j] = dmat[j, i] = np.linalg.norm(N_r[i] - N_r[j])

        while gamma > 0.1 * radius**3:
            gamma = self.__comp_gamma(t=t, radius=radius)
            for index in range(npts):
                m_i = np.zeros_like(N_r[index])
                for j in range(npts):
                    temp = N_r[j] - N_r[index]
                    norm = dmat[j, index]
                    m_i -= temp * gamma / norm**3
                if rtenth <= np.linalg.norm(m_i) <= radius:
                    hat_s_i = self.__hat_s_i(N_r[index], m_i)

                    if not np.array_equal(N_r[index], hat_s_i):
                        temp = self.__boule(radius=radius, N=N_r,
                                            center=hat_s_i, out=False)
                        if temp.shape[0] == 1:
                            N_r[index] = hat_s_i
                            for i in range(npts):
                                dmat[index, i] = dmat[i, index] = np.linalg.norm(N_r[index] - N_r[i])
                print(index)

            N = self.points.copy()

            for s_i in N_r:
                N = self.__boule(radius=radius, N=N, center=s_i)

            print(t)
            t += 1

        return np.array(N_r)

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


def taille_scan(scan):
    min_ = np.inf
    max_ = 0

    for row in scan:
        if row[2] < min_:
            min_ = row[2]
        if row[2] > max_:
            max_ = row[2]

    return max_ - min_


def normalize_scan(scan, size=170):
    scantemp = scan
    coeff = size / taille_scan(scan)
    scantemp *= coeff

    return(scantemp)


def decolored_dist(X1, X2):
    dist1 = 0
    dist2 = 0
    dist3 = 0
    if len(X1[0]) > 1 or len(X2[0]) > 1:
        dist1 = gudhi.wasserstein.wasserstein_distance(
            X1[0], X2[0], order=2, internal_p=2)
    if len(X1[1]) > 0 or len(X2[1]) > 0:
        dist2 = gudhi.wasserstein.wasserstein_distance(
            X1[1], X2[1], order=2, internal_p=2)
    if len(X1[2]) > 0 or len(X2[2]) > 0:
        dist3 = gudhi.wasserstein.wasserstein_distance(
            X1[2], X2[2], order=2, internal_p=2)

    return np.sqrt(dist1**2+dist2**2+dist3**2)


if __name__ == '__main__':
    pathglob = 'E:/TDA/Male/'
    all_scans = os.listdir(pathglob)

    def scan_from_index(i):
        path = pathglob + str(all_scans[i])
        scan = meshio.read(path)
        scan = scan.points
        return scan

    spl = np.random.choice(range(len(all_scans)), 2, replace=False)
    spl = [1487, 404]

    min_persistence = 3
    radius = np.sqrt(min_persistence)
    scan = scan_from_index(spl[0])
    A = normalize_scan(scan)
    PS = points_set(A)
    # PS.plot(title='Scan '+str(spl[0])+' - Original',
    #         color='green', opacity=0.5, size=2)

    B = PS.downsample(4500, radius)
    DS = points_set(B)
    # DS.plot(title='Scan '+str(spl[0])+' - Sous-échantillon algo.', opacity=0.5)

    # RS = points_set(np.array([A[i] for i in np.random.choice(
    #     range(A.shape[0]), B.shape[0], replace=False)]))
    # RS.plot(title='Scan '+str(spl[0])+' - Sous-échantillon aléa.')

    decolored_PS = PS.DecoloredPersistence(min_persistence=min_persistence)
    decolored_DS = DS.DecoloredPersistence(min_persistence=min_persistence)
    # decolored_RS = RS.DecoloredPersistence(min_persistence=min_persistence)

    print("SCAN "+str(spl[0]))
    print("Nombre de points :")
    print(" - Échantillon original :", A.shape[0])
    print(" - Sous-échantillons :", B.shape[0])
    print("Distance entre diagrammes :")
    print(" - Échantillon original et sous-échantillon ASAP :",
          decolored_dist(decolored_PS, decolored_DS))
    # print(" - Échantillon original et sous-échantillon aléatoire :",
    #       decolored_dist(decolored_PS, decolored_RS))

    # scan = scan_from_index(spl[1])
    # A = normalize_scan(scan)
    # PS2 = points_set(A)
    # # PS2.plot(title='Scan '+str(spl[1])+' - Original',
    # #         color='green', opacity=0.5, size=2)

    # B = PS2.downsample(radius)
    # DS2 = points_set(B)
    # # DS2.plot(title='Scan '+str(spl[1])+' - Sous-échantillon algo.')

    # RS2 = points_set(np.array([A[i] for i in np.random.choice(
    #     range(A.shape[0]), B.shape[0], replace=False)]))
    # # RS2.plot(title='Scan '+str(spl[1])+' - Sous-échantillon aléa.')

    # decolored_PS2 = PS2.DecoloredPersistence(min_persistence=min_persistence)
    # decolored_DS2 = DS2.DecoloredPersistence(min_persistence=min_persistence)
    # decolored_RS2 = RS2.DecoloredPersistence(min_persistence=min_persistence)

    # print("\nSCAN "+str(spl[1]))
    # print("Nombre de points :")
    # print(" - Échantillon original :", A.shape[0])
    # print(" - Sous-échantillons :", B.shape[0])
    # print("Distance entre diagrammes :")
    # print(" - Échantillon original et sous-échantillon obtenu par l'algorithme :",
    #       decolored_dist(decolored_PS2, decolored_DS2))
    # print(" - Échantillon original et sous-échantillon aléatoire :",
    #       decolored_dist(decolored_PS2, decolored_RS2))

    # print("\nSCANS "+str(spl[0])+" ET "+str(spl[1]))
    # print("Distance entre diagrammes :")
    # print(" - Échantillons originaux :",
    #       decolored_dist(decolored_PS, decolored_PS2))
    # print(" - Sous-échantillons obtenus par l'algorithme :",
    #       decolored_dist(decolored_DS, decolored_DS2))
    # print(" - Sous-échantillons aléatoires :",
    #       decolored_dist(decolored_RS, decolored_RS2))

    # fig = plt.figure(figsize=(12, 5))
    # fig.suptitle(str(spl[0])+' - Original')
    # PS.PlotPersistenceDiagram(axes=fig.add_subplot(121))
    # PS.PlotPersistenceBarcode(axes=fig.add_subplot(122))

    # fig = plt.figure(figsize=(12, 5))
    # fig.suptitle(str(spl[0])+' - Original')
    # PS.PlotPersistenceDiagram(axes=fig.add_subplot(121))
    # PS.PlotPersistenceBarcode(axes=fig.add_subplot(122))
    # fig.savefig("E:/Projet-M2/2021-m2-ditex-morphotypes/Images/asap_scan_a.jpg",
    #             bbox_inches='tight', dpi=150)

    # fig = plt.figure(figsize=(12, 5))
    # fig.suptitle(str(spl[0])+' - Downsampled')
    # DS.PlotPersistenceDiagram(axes=fig.add_subplot(121))
    # DS.PlotPersistenceBarcode(axes=fig.add_subplot(122))
    # fig.savefig("E:/Projet-M2/2021-m2-ditex-morphotypes/Images/asap_scan_b.jpg",
    #             bbox_inches='tight', dpi=150)

    # fig = plt.figure(figsize=(12, 5))
    # fig.suptitle(str(spl[0])+' - Random')
    # RS.PlotPersistenceDiagram(axes=fig.add_subplot(121))
    # RS.PlotPersistenceBarcode(axes=fig.add_subplot(122))
    # fig.savefig("E:/Projet-M2/2021-m2-ditex-morphotypes/Images/asap_scan_c.jpg",
    #             bbox_inches='tight', dpi=150)

    # fig = plt.figure(figsize=(12, 5))
    # fig.suptitle(str(spl[1])+' - Original')
    # PS2.PlotPersistenceDiagram(axes=fig.add_subplot(121))
    # PS2.PlotPersistenceBarcode(axes=fig.add_subplot(122))

    # fig = plt.figure(figsize=(12, 5))
    # fig.suptitle(str(spl[1])+' - Downsampled')
    # DS2.PlotPersistenceDiagram(axes=fig.add_subplot(121))
    # DS2.PlotPersistenceBarcode(axes=fig.add_subplot(122))
    # plt.show()
