import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial

data = pd.read_csv("../data/table_7.4.csv")
data = data.values


def least_squares_quadratic(x, y):
    C = np.c_[np.ones(x.shape), x, x**2]
    beta = np.linalg.inv(C.T @ C) @ C.T @ y
    return beta, C, beta @ C.T


def loess(x, y, alpha=0.3):
    N = len(x)
    frac = int(alpha * N)
    diffs = abs(x - x[None, :].T)
    neighbors = np.argsort(diffs, axis=1)[:, :frac]
    rzs = []
    for d, n, _x in zip(diffs, neighbors, x):
        u = d[n] / np.max(d[n])
        w = (1 - u**3) ** 3
        xn = np.c_[np.ones(x[n].shape), x[n]]
        xp = np.diag(w) @ xn
        yp = np.diag(w) @ y[n]
        beta = np.linalg.inv(xp.T @ xp) @ xp.T @ yp
        rz = beta @ np.c_[1, _x].T
        rzs.append(rz)
    return np.array(rzs)


yloess = loess(data[:, 0], data[:, 1])

beta, C, yls = least_squares_quadratic(data[:, 0], data[:, 1])

plt.scatter(data[:, 0], data[:, 1], marker="+", c="k")
plt.plot(data[:, 0], yls, c="k", lw=2, ls="dashed")
plt.plot(data[:, 0], yloess, c="k", lw=2)
plt.show()


def r_lstsq_fn(data, xstar):
    beta, _, _ = least_squares_quadratic(data[:, 0], data[:, 1])
    xstar = np.c_[1, xstar, xstar**2]
    return np.squeeze(beta @ xstar.T)


x60_fn = partial(r_lstsq_fn, xstar=60)
x100_fn = partial(r_lstsq_fn, xstar=100)


x60_bstderr = bootstrap.bootstrap_stderr(
    data,
    stat_fn=x60_fn,
    n_boots=250,
)

x100_bstderr = bootstrap.bootstrap_stderr(
    data,
    stat_fn=x100_fn,
    n_boots=250,
)

r_lstsq_60 = r_lstsq_fn(data, xstar=60)
r_lstsq_100 = r_lstsq_fn(data, xstar=100)
print(r_lstsq_60, x60_bstderr)
print(r_lstsq_100, x100_bstderr)
