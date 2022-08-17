import utils as ut
import multiprocessing as mp
from time import time

argparse = ut.importer('argparse')
read = ut.importer('meshio', 'read')
dump, load = ut.importer('pickle', 'dump', 'load')
timedelta = ut.importer('datetime', 'timedelta')
loadmat = ut.importer('scipy.io', 'loadmat')


def diagrams(ps, min_p):
    ps.Persistence(min_persistence=min_p)
    diag_decolor = ps.DecoloredPersistence()

    return diag_decolor


if __name__ == '__main__':
    nb_cpu = mp.cpu_count()
    min_persistence = 5500 / 999

    print("\033[H\033[JMALE : ", end="")
    print('Loading data...')

    pool = mp.Pool(nb_cpu)
    with open('./Data/male.bin', 'rb') as fp:
        psets = load(fp)

    sets = [ut.point_set(pset) for pset in psets]
    pool.close()

    begin = time()
    print("\033[H\033[JMALE : ", end="")
    print('Computing diagrams...')
    k = 1

    pool = mp.Pool(nb_cpu)
    diags = pool.starmap(diagrams, [(set, min_persistence) for set in sets])
    pool.close()

    Xd = list(diags)

    del sets

    print('\rSaving diagrams       ', end='')

    with open('.Data/XY1_trunk_decolor.bin', 'wb') as fp:
        dump(Xd, fp)

    print("\033[H\033[JFEMALE : ", end="")
    print('Loading data...')

    pool = mp.Pool(nb_cpu)
    with open('./Data/female.bin', 'rb') as fp:
        psets = load(fp)

    sets = [ut.point_set(pset) for pset in psets]
    pool.close()

    begin = time()
    print("\033[H\033[JFEMALE : ", end="")
    print('Computing diagrams...')
    k = 1

    pool = mp.Pool(nb_cpu)
    diags = pool.starmap(diagrams, [(set, min_persistence) for set in sets])
    pool.close()

    Xd = list(diags)

    del sets

    print('\rSaving diagrams       ', end='')

    with open('./Data/XX_trunk_decolor.bin', 'wb') as fp:
        dump(Xd, fp)

    print('\rFinished!             ')
