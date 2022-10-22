import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial

data = np.random.normal(size=10)


def stat_fn(data):
    return np.exp(np.mean(data))


orig_stat = stat_fn(data)

bstat = bootstrap.bootstrap_stat(data, stat_fn)
alpha = 0.025
lo = np.quantile(bstat, q=alpha)
up = np.quantile(bstat, q=1 - alpha)

lo_std, up_std = bootstrap.bootstrap_standard_ci(
    data,
    stat_fn=stat_fn,
    alpha=alpha,
)

# to test if it works
lo_pci, up_pci = bootstrap.bootstrap_percentile_ci(
    data,
    stat_fn=stat_fn,
    alpha=alpha,
)


def plot():
    plt.hist(bstat, bins=20, ec="k", fc="w")
    plt.axvline(orig_stat, lw=2, c="blue")
    plt.axvline(lo, ls="dashed", c="k")
    plt.axvline(up, ls="dashed", c="k")
    plt.axvline(lo_std, ls="dashed", c="orange")
    plt.axvline(up_std, ls="dashed", c="orange")
    plt.show()


plot()
