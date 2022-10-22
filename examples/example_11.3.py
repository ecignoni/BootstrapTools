import sys

sys.path.append("../")
import jackknife

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial

data = pd.read_csv("../data/table_7.1.csv")
data = data.values


def ratio_largest_eigenvalue(data):
    G = np.cov(data, rowvar=False, ddof=0)
    Gvals, Gvecs = np.linalg.eigh(G)
    return Gvals[-1] / np.sum(Gvals)


jstat = jackknife.jackknife_stat(data, ratio_largest_eigenvalue)


def plot_histogram():
    mod_jstat = (len(jstat) - 1) ** 0.5 * (jstat - np.mean(jstat))
    plt.hist(mod_jstat + np.mean(jstat), bins=20, ec="k", fc="w")
    # plt.hist(jstat, bins=20, ec='k', fc='w')
    plt.show()


# plot_histogram()

jstderr = jackknife.jackknife_stderr(data, ratio_largest_eigenvalue)

print(jstderr)
