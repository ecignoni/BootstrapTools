import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial

data = pd.read_csv("../data/table_8.1.csv")
data = data.values

bdata = bootstrap.bootstrap_moving_block(data[:, 1], block=5, n_boots=200)


def arm1(y):
    zt = (y - np.mean(y))[1:].reshape(-1, 1)
    zt_1 = (y - np.mean(y))[:-1].reshape(-1, 1)
    b = np.linalg.inv(zt_1.T @ zt_1) @ zt_1.T @ zt
    return b[0][0]


bstats = np.array([arm1(bd) for bd in bdata])
bstderr = np.std(bstats, ddof=1)
print(bstderr)
