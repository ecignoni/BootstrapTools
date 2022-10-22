"""
Example 6.3 from the book of Efron:
Efron, Bradley, and Robert J. Tibshirani. An introduction to the bootstrap. CRC press, 1994.
"""

import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../data/table_3.1.csv")

# Our stat_fn
def corr_fn(data):
    return np.corrcoef(data[:, 0], data[:, 1])[0, 1]


# Get the data in a numpy array
data = data[["LSAT", "GPA"]].values

# Correlation in the original data
corr_data = corr_fn(data)

n_boots_trials = [25, 50, 100, 200, 400, 800, 1600, 3200]

# Compute the bootstrap standard error for each
# n_boots
stderr_bs = []
for n_boots in n_boots_trials:
    stderr_b = bootstrap.bootstrap_stderr(data, stat_fn=corr_fn, n_boots=n_boots)
    stderr_bs.append(stderr_b)

#plt.plot(n_boots_trials, stderr_bs)
#plt.show()

# Get a bootstrap sample of the statistics
bstats = bootstrap.bootstrap_stat(data, stat_fn=corr_fn, n_boots=3200)

# Histogram of the statistics is non-normal
#plt.hist(bstats, bins=49)
#plt.show()
