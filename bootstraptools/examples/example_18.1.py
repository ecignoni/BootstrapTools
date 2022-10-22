import sys
sys.path.append('../')
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
from functools import partial

data = pd.read_csv('../data/table_18.1.csv')
data = data.values

age = data[:, 1].astype(float)
logC = data[:, 2].astype(float)

idxsort = np.argsort(age)
age = age[idxsort]
logC = logC[idxsort]

spl = UnivariateSpline(age, logC, k=3, s=14)

def plot():
    plt.scatter(age, logC, ec='k', fc='w')
    plt.plot(age, spl(age), c='k')
    plt.show()

#plot()

def loss_fn(x, lambd, a_orig, b_orig):
    a, b = x[:, 0], x[:, 1]
    idxsort = np.argsort(a)
    a = a[idxsort]
    b = b[idxsort]
    spl = UnivariateSpline(a, b, k=3, s=lambd)
    return np.mean((b_orig - spl(a_orig))**2)

x = np.column_stack((age, logC))

lambda_grid = np.arange(5, 20, 1)
bootstrap_errors = []
for lambd in lambda_grid:
    stat_fn = partial(loss_fn, lambd=lambd, a_orig=age, b_orig=logC)
    bpse = bootstrap.bootstrap_stat(x, stat_fn=stat_fn, n_boots=100)
    bpse = np.mean(bpse[~np.isnan(bpse)])
    bootstrap_errors.append(bpse)

def plot():
    plt.plot(lambda_grid, bootstrap_errors, '-o', c='k')
    plt.show()

plot()
