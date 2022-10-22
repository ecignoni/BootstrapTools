import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial

data = pd.read_csv("../data/table_2.1.csv")
data = data.values

labels, data = data[:, 0], data[:, 1]
control = data[labels == "C"]
treatment = data[labels == "T"]

conf_norm = bootstrap.bootstrap_standard_ci(
    control,
    stat_fn=np.mean,
    alpha=0.05,
)

conf_t = bootstrap.bootstrap_studentt_ci(
    control,
    stat_fn=np.mean,
    alpha=0.05,
)

conf_bt = bootstrap.bootstrap_t_ci(
    control,
    stat_fn=np.mean,
    alpha=0.05,
)

print("[{:.2f},{:.2f}]".format(conf_norm[0], conf_norm[1]))
print("[{:.2f},{:.2f}]".format(conf_t[0], conf_t[1]))
print("[{:.2f},{:.2f}]".format(conf_bt[0], conf_bt[1]))
