import sys

sys.path.append("../")
import bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from functools import partial

data = pd.read_csv("../data/table_9.1.csv")
data = data.values

labels = data[:, 0].astype(str)
hours = data[:, 1].astype(float)
amount = data[:, 2].astype(float)


class OLS:
    def __init__(self):
        pass

    def fit(self, x, y):
        c = np.c_[np.ones(x.shape), x]
        self.beta = np.linalg.inv(c.T @ c) @ c.T @ y
        return self

    def predict(self, x):
        c = np.c_[np.ones(x.shape), x]
        return self.beta @ c.T


ols = OLS().fit(hours, amount)


def RSEn(y, yhat):
    return np.sum((y - yhat) ** 2) / len(y)


skols = LinearRegression().fit(hours.reshape(-1, 1), amount.reshape(-1, 1))


print(bootstrap.bootstrap_632_error(hours, amount, ols, RSEn))
