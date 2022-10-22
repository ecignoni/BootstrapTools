import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial

data = pd.read_csv("../data/table_3.1.csv")
data = data[["LSAT", "GPA"]].values

mean = np.mean(data, axis=0)
cov = np.cov(data, rowvar=False, ddof=1)

distrib_fn = partial(
    np.random.multivariate_normal,
    mean=mean,
    cov=cov,
)


def corr_fn(data):
    return np.corrcoef(data[:, 0], data[:, 1])[0, 1]


bdata = bootstrap.bootstrap_parametric_sample(
    distrib_fn,
    size=data.shape[0],
    n_boots=10000,
)

bstats = bootstrap.bootstrap_parametric_stat(
    distrib_fn,
    size=data.shape[0],
    stat_fn=corr_fn,
    n_boots=10000,
)

plt.hist(bstats, bins=49, ec="w", fc="orange")
plt.show()

bstderr = bootstrap.bootstrap_parametric_stderr(
    distrib_fn,
    size=data.shape[0],
    stat_fn=corr_fn,
    n_boots=10000,
)

print(bstderr)
