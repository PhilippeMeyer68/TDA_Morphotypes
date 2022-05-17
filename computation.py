# Depuis le répertoire du dépôt:
#   - Pour calculer les diagrammes et les distances entre eux pour les hommes:
#       python .\computation.py -i [Chemin du répertoire contenant les scans masculins] -o ./Data -f XX -v
#   - Pour calculer les diagrammes et les distances entre eux pour les femmes:
#       python .\computation.py -i [Chemin du répertoire contenant les scans féminins] -o ./Data -f XX -v
#   - Pour calculer les distances entre tous les diagrammes:
#       python .\computation.py -i ./Data/XY -i ./Data/XX -o ./Data -f X -v -d

import argparse
from logging import warning
import utils
from os import listdir, makedirs
from os.path import exists
from meshio import read
from pickle import dump, load

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Computation', description='Compute diagrams and distances')
    parser.add_argument('-m', '--min_persistence', type=float, default=3.)
    parser.add_argument('-d', '--dist', action='store_true', help='Compute only distance (Diagrams files must exist !)')
    parser.add_argument('-i', '--input', action='append',
                        help='<Required> Input directory (+ root of files if --dist)', required=True)
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('-f', '--file_name', help='Root of output files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Displays the remaining calculation time')
    args = parser.parse_args()

    all_scans = []
    nb_scans = []
    nb_total = 0

    def scan_from_index(i, index):
        path = args.input[i] + str(all_scans[i][index])
        scan = read(path)
        scan = scan.points
        return scan

    if args.verbose == True:
        from time import time
        from numpy import round

    X = []
    X_decolor = []

    print("\033[H\033[J", end="")
    print(args.input, '-->', args.output)

    if args.output[-1] != '\\' and args.output[-1] != '/':
        args.output += '/'
    
    if not exists(args.output):
        makedirs(args.output)

    if args.file_name == None:
        args.file_name = 'X'

    if not args.dist:
        for i in range(len(args.input)):
            if args.input[i][-1] != '\\' and args.input[i][-1] != '/':
                args.input[i] += '/'
            temp = listdir(args.input[i])
            all_scans.append(temp)
            length = len(temp)
            nb_scans.append(length)
            nb_total += length

        if args.verbose == True:
            begin = time()
            k = 1

        for i in range(len(args.input)):
            for index in range(nb_scans[i]):
                scan = scan_from_index(i, index)

                points_set = utils.points_set(scan)
                points_set.normalize()
                diag = points_set.Persistence(
                    min_persistence=args.min_persistence)
                X.append(diag)

                diag_decolor = points_set.DecoloredPersistence()
                X_decolor.append(diag_decolor)

                if args.verbose:
                    temp = (time()-begin) / k * (nb_total-k)
                    h = int(temp // 3600)
                    temp = temp - h * 3600
                    m = int(temp // 60)
                    s = int(temp % 60)
                    temps_restant = ''
                    if h > 0:
                        temps_restant = str(h)+'h'+str(m)+'m'+str(s)+'s'
                    elif m > 0:
                        temps_restant = str(m)+'m'+str(s)+'s'
                    else:
                        temps_restant = str(s)+'s'
                    print('\rCalcul des diagrammes : ' +
                          temps_restant + '       ', end='')
                    k += 1

        print('\rSaving diagrams       ', end='')

        with open(args.output+args.file_name+'.bin', 'wb') as fp:
            dump(X, fp)

        with open(args.output+args.file_name+'_decolor.bin', 'wb') as fp:
            dump(X_decolor, fp)
    else:
        for npt in args.input:
            with open(npt+'.bin', 'rb') as fp:
                X += load(fp)

            with open(npt+'_decolor.bin', 'rb') as fp:
                X_decolor += load(fp)
        nb_total = len(X)

    dict_dist_0 = {}
    dict_dist_1 = {}
    dict_dist_2 = {}

    if args.verbose == True:
        begin = time()
        n = nb_total * (nb_total - 1) / 2

    for i in range(nb_total):
        for j in range(i+1, nb_total):
            dict_dist_0[(i, j)], dict_dist_1[(i, j)], dict_dist_2[(
                i, j)] = utils.decolored_dist(X_decolor[i], X_decolor[j], [0, 1, 2])
            if args.verbose:
                m = (nb_total - i) * (nb_total - i - 1) / 2 + j - i
                if m % (n // 200) == 0:
                    temp = (time()-begin) / (n - m) * (m - 1)
                    h = int(temp // 3600)
                    temp = temp - h * 3600
                    m = int(temp // 60)
                    s = int(temp % 60)
                    temps_restant = ''
                    if h > 0:
                        temps_restant = "{:02d}".format(
                            h)+'h'+"{:02d}".format(m)+'m'+"{:02d}".format(s)+'s'
                    elif m > 0:
                        temps_restant = "{:02d}".format(
                            m)+'m'+"{:02d}".format(s)+'s'
                    else:
                        temps_restant = "{:02d}".format(s)+'s'
                    print('\rCalcul des distances : ' +
                          temps_restant + '       ', end='')
    print('\rSaving distances       ', end='')

    with open(args.output+args.file_name+'_Dist_H0.bin', 'wb') as fp:
        dump(dict_dist_0, fp)
    with open(args.output+args.file_name+'_Dist_H1.bin', 'wb') as fp:
        dump(dict_dist_1, fp)
    with open(args.output+args.file_name+'_Dist_H2.bin', 'wb') as fp:
        dump(dict_dist_2, fp)
    print('\rFinished       ', end='')
