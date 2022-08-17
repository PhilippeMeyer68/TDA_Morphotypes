# Depuis le répertoire du dépôt:
#   - Pour calculer les diagrammes pour les hommes:
#       python .\computation.py -i [Chemin du répertoire contenant les scans masculins] -o ./Data -f XX -v -diag
#   - Pour calculer les diagrammes et les distances entre eux pour les femmes:
#       python .\computation.py -i [Chemin du répertoire contenant les scans féminins] -o ./Data -f XX -v -diag -dist
#   - Pour calculer les distances entre tous les diagrammes:
#       python .\computation.py -i ./Data/XY.bin -i ./Data/XX.bin -o ./Data -f X -v -dist -p

import utils as ut
from os import listdir, makedirs
from os.path import exists

argparse = ut.importer('argparse')
read = ut.importer('meshio', 'read')
dump, load = ut.importer('pickle', 'dump', 'load')
timedelta = ut.importer('datetime', 'timedelta')
loadmat = ut.importer('scipy.io', 'loadmat')


def scan_from_index(path):
    if path[-4:] == '.mat':
        mat = loadmat(path)
        scan = mat['points']
    else:
        scan = read(path)
        scan = scan.points
    ps = ut.point_set(scan)
    ps.normalize()

    return ps


def diagrams(ps, min_p):
    diag = ps.Persistence(min_persistence=min_p)
    diag_decolor = ps.DecoloredPersistence()

    return (diag, diag_decolor)


def decolored_dist(Xi, Xj, p, order):
    return ut.decolored_dist(Xi, Xj, [0, 1, 2], p=p, order=order)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Computation', description='Compute diagrams and distances')
    parser.add_argument('-wp', '--internal_p', type=float,
                        default=2., help='Wasserstein internal p')
    parser.add_argument('-wo', '--order', type=float,
                        default=2., help='Wasserstein order')
    parser.add_argument('-mp', '--min_persistence', type=float,
                        default=3., help='Diagrams minimum persistence')
    parser.add_argument('-dist', '--distances', action='store_true',
                        help='Compute distances')
    parser.add_argument('-diag', '--diagrams', action='store_true',
                        help='Compute diagrams')
    parser.add_argument('-i', '--input', action='append',
                        help='If -diags: Input directory\nElse     : Input file', required=True)
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('-f', '--file_name', help='Root of output files')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='Allow parallelisation')
    args = parser.parse_args()

    all_scans = []
    nb_scans = []
    nb_total = 0

    if not args.parallel:
        def scan_from_index(i, index):
            path = args.input[i] + str(all_scans[i][index])
            if path[-4:] == '.mat':
                mat = loadmat(path)
                scan = mat['points']
            else:
                scan = read(path)
                scan = scan.points
            return scan
    else:
        import multiprocessing as mp
        nb_cpu = mp.cpu_count()

    if args.verbose == True:
        from time import time

    X = []
    Xd = []

    print("\033[H\033[J", end="")
    print(args.input, '-->', args.output)

    if args.output[-1] != '\\' and args.output[-1] != '/':
        args.output += '/'

    if not exists(args.output):
        makedirs(args.output)

    if args.file_name == None:
        args.file_name = 'X'

    if args.diagrams:
        for i in range(len(args.input)):
            if args.input[i][-1] != '\\' and args.input[i][-1] != '/':
                args.input[i] += '/'
            temp = listdir(args.input[i])
            all_scans.append(temp)
            length = len(temp)
            nb_scans.append(length)
            nb_total += length

        if args.parallel:
            for i in range(len(args.input)):
                if args.verbose:
                    print('Loading data...')

                paths = [args.input[i]+all_scans[i][index]
                         for index in range(nb_scans[i])]

                pool = mp.Pool(nb_cpu)
                sets = pool.map(scan_from_index, paths)
                pool.close()

                if args.verbose == True and i == 0:
                    begin = time()
                    print("\033[H\033[J", end="")
                    print(args.input, '-->', args.output+args.file_name)
                    print('Computing diagrams...')
                    k = 1

                pool = mp.Pool(nb_cpu)
                diags = pool.starmap(
                    diagrams, [(set, args.min_persistence) for set in sets])
                pool.close()

                for diag in diags:
                    X.append(diag[0])
                    Xd.append(diag[1])

                del sets
        else:
            if args.verbose == True:
                begin = time()
                print("\033[H\033[J", end="")
                print(args.input, '-->', args.output+args.file_name)
                print('Computing diagrams')
                k = 1

            for i in range(len(args.input)):
                for index in range(nb_scans[i]):
                    scan = scan_from_index(i, index)

                    point_set = ut.point_set(scan)
                    point_set.normalize()
                    diag = point_set.Persistence(min_persistence=3)
                    X.append(diag)

                    diag_decolor = point_set.DecoloredPersistence()
                    Xd.append(diag_decolor)

                    if args.verbose:
                        temp = (time()-begin) / k * (nb_total-k)
                        temps_restant = str(timedelta(seconds=temp))[:7]
                        print('\r   Estimated remaining time : ' +
                              temps_restant, end='')
                        k += 1

        print('\rSaving diagrams       ', end='')

        with open(args.output+args.file_name+'.bin', 'wb') as fp:
            dump(X, fp)

        with open(args.output+args.file_name+'_decolor.bin', 'wb') as fp:
            dump(Xd, fp)
    else:
        for input_file in args.input:
            with open(input_file, 'rb') as fp:
                Xd += load(fp)
        nb_total = len(Xd)

    if args.distances:
        d0 = {}
        d1 = {}
        d2 = {}

        if args.parallel:
            if args.verbose == True:
                begin = time()
                print("\033[H\033[J", end="")
                print(args.input, '-->', args.output+args.file_name)
                print('Computing distances...')

            pool = mp.Pool(nb_cpu)
            for i in range(nb_total):
                temp = pool.starmap(
                    decolored_dist, [(Xd[i], Xj, args.internal_p, args.order) for Xj in Xd[i+1:]])
                for j in range(i+1, nb_total):
                    [d0[(i, j)], d1[(i, j)], d2[(i, j)]] = temp[j-i-1]
            pool.close()
        else:
            if args.verbose == True:
                begin = time()
                print("\033[H\033[J", end="")
                print(args.input, '-->', args.output+args.file_name)
                print('Computing distances')
                n = nb_total * (nb_total - 1) / 2
                k = 1

            for i in range(nb_total):
                for j in range(i+1, nb_total):
                    [d0[(i, j)], d1[(i, j)], d2[(i, j)]] = decolored_dist(
                        Xd[i], Xd[j], p=args.internal_p, order=args.order)
                    if args.verbose:
                        if k % nb_total == 0:
                            temp = (time()-begin) / k * (n - k)
                            temps_restant = str(timedelta(seconds=temp))[:7]
                            print('\r   Estimated remaining time : ' +
                                  temps_restant, end='')
                        k += 1

        if args.verbose == True:
            print("\033[H\033[J", end="")
            print(args.input, '-->', args.output+args.file_name)
            print('\rSaving distances...', end='')

        with open(args.output+args.file_name+'_Dist_H0.bin', 'wb') as fp:
            dump(d0, fp)
        with open(args.output+args.file_name+'_Dist_H1.bin', 'wb') as fp:
            dump(d1, fp)
        with open(args.output+args.file_name+'_Dist_H2.bin', 'wb') as fp:
            dump(d2, fp)
    print('\rFinished!             ')
