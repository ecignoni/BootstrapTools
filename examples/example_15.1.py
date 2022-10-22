import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp_stats

from functools import partial

data = pd.read_csv("../data/table_2.1.csv")
data = data.values

labels = data[:, 0]
values = data[:, 1]

C = values[labels == "C"].astype(float)
T = values[labels == "T"].astype(float)


def stat_fn(C, T):
    return np.mean(T) - np.mean(C)


orig_stat = stat_fn(C, T)
print("original statistics: {:.2f}".format(orig_stat))


def ztest_ind(a, b):
    n = len(a)
    m = len(b)
    theta = np.mean(a) - np.mean(b)
    std = (
        (np.sum((a - np.mean(a)) ** 2) + np.sum((b - np.mean(b)) ** 2)) / (n + m - 2)
    ) ** 0.5
    mstd = std * (1 / n + 1 / m) ** 0.5
    asl = 1 - sp_stats.norm.cdf(theta / mstd)
    return asl


ztest_asl = ztest_ind(T, C)
print("ASL for z-test:", ztest_asl)

ttest_asl = sp_stats.ttest_ind(T, C, alternative="greater").pvalue
print("ASL for t-test:", ttest_asl)

permtest_asl = sp_stats.permutation_test((C, T), stat_fn, alternative="greater").pvalue
print("ASL for permutation test:", permtest_asl)

bootstrap_asl = bootstrap.bootstrap_test(
    C, T, stat_fn, alternative="greater", progbar=False
)
print("ASL for bootstrap test:", bootstrap_asl)

asl = bootstrap.bootstrap_test_equalmeans(C, T, progbar=False)
print(asl)
