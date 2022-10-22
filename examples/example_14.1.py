import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial

data = pd.read_csv("../data/table_14.1.csv")
data = data.values

A, B = data[:, 0], data[:, 1]

# out stat fn is a biased variance
stat_fn = partial(np.var, ddof=0)

A_stat = stat_fn(A)

A_bstat = bootstrap.bootstrap_stat(
    A,
    stat_fn=stat_fn,
    n_boots=2000,
)

n_boots = 2000
alpha = 0.05

lo_std, up_std = bootstrap.bootstrap_standard_ci(
    A,
    stat_fn,
    n_boots=n_boots,
    alpha=alpha,
    progbar=False,
)

lo_pci, up_pci = bootstrap.bootstrap_percentile_ci(
    A,
    stat_fn,
    n_boots=n_boots,
    alpha=alpha,
    progbar=False,
)

lo_btci, up_btci = bootstrap.bootstrap_t_ci(
    A,
    stat_fn,
    n_boots=n_boots,
    alpha=alpha,
    progbar=False,
)

lo_bca, up_bca = bootstrap.bootstrap_bca_ci(
    A,
    stat_fn,
    n_boots=n_boots,
    alpha=alpha,
    progbar=False,
)

print("standard", lo_std, up_std)
print("percentile", lo_pci, up_pci)
print("bootstrap-t", lo_btci, up_btci)
print("BCa", lo_bca, up_bca)


def plot():
    plt.hist(A_bstat, bins=20, ec="k", fc="w")
    plt.show()


# plot()
