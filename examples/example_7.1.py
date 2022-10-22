import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial

data = pd.read_csv("../data/table_7.1.csv")
data = data.values

x_bar = np.mean(data, axis=0)
print("x bar:", x_bar)
G = np.cov(data, rowvar=False, ddof=0)
print("G:", G)
Gvals, Gvecs = np.linalg.eigh(G)
Gvals = Gvals[::-1]
print("G eigenvalues:", Gvals)
Gvecs = Gvecs[:, ::-1]
print("G eigenvectors:", Gvecs)


def highest_eval_ratio(data):
    G = np.cov(data, rowvar=False, ddof=0)
    Gvals, Gvecs = np.linalg.eigh(G)
    return Gvals[-1] / np.sum(Gvals)


orig_stat = highest_eval_ratio(data)

bstats = bootstrap.bootstrap_stat(
    data,
    stat_fn=highest_eval_ratio,
    n_boots=10000,
)

plt.hist(bstats, bins=49, ec="w", fc="orange")
plt.axvline(orig_stat, ls="dashed", c="blue", lw=3)
plt.show()

bstderr = bootstrap.bootstrap_stderr(
    data,
    stat_fn=highest_eval_ratio,
    n_boots=10000,
)

print(bstderr)


def principal_component(data, pc=0):
    G = np.cov(data, rowvar=False, ddof=0)
    Gvals, Gvecs = np.linalg.eigh(G)
    Gvals = Gvals[::-1]
    Gvecs = Gvecs[:, ::-1]
    if np.all(Gvecs[:, -1] < 0):
        Gvecs *= -1
    return Gvecs[:, pc]


pc0_fn = partial(principal_component, pc=0)
pc1_fn = partial(principal_component, pc=1)

bstderr_pc0 = bootstrap.bootstrap_stderr(
    data,
    stat_fn=pc0_fn,
    n_boots=1000,
)
bstderr_pc1 = bootstrap.bootstrap_stderr(
    data,
    stat_fn=pc1_fn,
    n_boots=1000,
)

print(bstderr_pc0)
print(bstderr_pc1)
