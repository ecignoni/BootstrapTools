import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t

from functools import partial

data = pd.read_csv("../data/table_2.1.csv")
data = data.values

labels, values = data[:, 0], data[:, 1]

T = values[labels == "T"].astype(float)
C = values[labels == "C"].astype(float)

asl_normal = norm.cdf((np.mean(T) - 129.0) * len(T) ** 0.5 / (np.std(T, ddof=1)))
print(asl_normal)
