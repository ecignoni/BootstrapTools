import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial

data = pd.read_csv("../data/table_8.1.csv")
data = data.values


def plot_data(data):
    plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(data[:, 0], data[:, 1], lw=2, c="k")
    plt.axhline(np.mean(data[:, 1]), ls="dashed", c="k")
    plt.show()


# plot_data(data)


def arm1(t, y):
    zt = (y - np.mean(y))[1:].reshape(-1, 1)
    zt_1 = (y - np.mean(y))[:-1].reshape(-1, 1)
    b = np.linalg.inv(zt_1.T @ zt_1) @ zt_1.T @ zt
    return b, np.squeeze(zt.T - b @ zt_1.T)


b, eps = arm1(data[:, 0], data[:, 1])


def plot_disturbances(eps):
    plt.hist(eps, ec="w", fc="orange")
    plt.show()


# plot_disturbances(eps)
# print(np.mean(eps))
# print(np.std(eps, ddof=1))


def bootstrap_arm1(y, b, eps, n_boots=1000):
    z = y - np.mean(y)
    zt = z[1:]
    bzs = []
    for _ in range(n_boots):
        beps = np.random.choice(eps, size=len(eps), replace=True)
        bz = [z[0]]
        for e in beps:
            bz.append(b * bz[-1] + e)
        bz = np.array(bz)
        bzs.append(bz)
    return np.array(bzs)


bzs = bootstrap_arm1(data[:, 1], b[0][0], eps, n_boots=100)


def plot_bsamples(bzs):
    for bz in bzs:
        plt.plot(bz, lw=0.33)
    plt.show()
