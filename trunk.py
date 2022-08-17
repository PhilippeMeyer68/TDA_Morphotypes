import numpy as np
import meshio
import os
from pickle import dump


def body_to_trunk(points):
    points = points[np.argsort(points[:, 2])]
    G = points.mean(axis=0)
    G[2] = points[0, 2]
    to_plot = (points - G).T
    taille = to_plot[2, -1]

    to_plot *= 170 / taille
    to_plot[0] -= 0.7

    b = 0.8192319205190405
    a = -0.5734623443633283

    to_plot = to_plot[:, :10990]
    temp = to_plot.copy()
    to_plot[0] = temp[0]*a + temp[1]*b
    to_plot[1] = temp[1]*a - temp[0]*b
    del temp

    crotch = to_plot[2, -1] - 80
    boolArr = (to_plot[2] > crotch)
    x = np.max(np.abs(to_plot[0, 10400:]))
    to_plot = to_plot[:, boolArr]
    to_plot[2] -= np.min(to_plot[2])

    X = np.abs(to_plot[0])
    a = x * (1 + (70 - to_plot[2]) * 0.001)
    X = np.abs(to_plot[0])

    boolArr = (X < a)
    temp = (to_plot[:, boolArr]).copy()
    boolArr = temp[2] < 30
    temp = temp[:, boolArr]
    b = np.max(temp[0]**2+temp[1]**2)**.5

    a = (40 - to_plot[2]) * (b - x) / 30 + b
    boolArr = (X < a) + ((X < b) & (to_plot[1] < 40))
    to_plot = to_plot[:, boolArr]

    return to_plot.T

if __name__ == '__main__':
    path = 'E:/TDA/Male/'
    files = os.listdir(path)
    points_arr = []

    for file in files:
        scan = meshio.read(path+file)
        points = scan.points

        points_arr.append(body_to_trunk(points=points))

    with open('E:/TDA/male.bin', 'wb') as fp:
        dump(points_arr, fp)

    path = 'E:/TDA/Female/'
    files = os.listdir(path)
    points_arr = []

    for file in files:
        scan = meshio.read(path+file)
        points = scan.points

        points_arr.append(body_to_trunk(points=points))

    with open('E:/TDA/female.bin', 'wb') as fp:
        dump(points_arr, fp)
