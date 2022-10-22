import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../data/table_2.1.csv")
labels = data["group"].values
data = data["data"].values


def diff_mean(*data):
    group1 = data[0]
    group2 = data[1]
    return np.mean(group1) - np.mean(group2)


orig_stat = diff_mean(data[labels == "T"], data[labels == "C"])

bstats = bootstrap.bootstrap_n_stat(
    data[labels == "T"],
    data[labels == "C"],
    stat_fn=diff_mean,
)

plt.hist(bstats, bins=49, fc="orange", ec="w")
plt.axvline(orig_stat, lw=3, ls="dashed", c="blue")
plt.show()

bstderr = bootstrap.bootstrap_n_stderr(
    data[labels == "T"],
    data[labels == "C"],
    stat_fn=diff_mean,
)

print(bstderr)
