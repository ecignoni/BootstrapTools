"""
Collection of functions to perform the bootstrap.
More details can be found in Ref. [1].

[1] Efron, Bradley, and Robert J. Tibshirani.
    An introduction to the bootstrap. CRC press, 1994.
"""

import numpy as np
from functools import partial
from scipy import stats as sp_stats

try:
    from tqdm import tqdm

    verb_iter = partial(
        tqdm, bar_format="{l_bar}{bar:20}|({rate_fmt}{postfix})", desc="Bootstrapping"
    )
except ModuleNotFoundError:

    class verbose_iterator:
        def __init__(self, iterator):
            self.orig_iterator = iterator
            self.iterator = enumerate(iterator)

        def __iter__(self):
            return self

        def __next__(self):
            try:
                i, it = next(self.iterator)
                print("Bootstrapping: {:d}".format(i), end="\r")
                return it
            except StopIteration as e:
                print()
                raise e

    verb_iter = verbose_iterator


__author__ = "Edoardo Cignoni"


# ============================================================================
# Jackknife-related functions
# ============================================================================
# These functions are in this mini-module for two reasons:
# 1. The jackknife is an approximation to the bootstrap
# 2. The jackknife is sometimes used to compute quantities within bootstrap
#    procedures (e.g., estimating the acceleration factor in the BCa method)


def jackknife_sample(data):
    """
    Performs a jackknife sampling of data.
    See Chapter 11 of Ref [1].
    Params
    ------
    data      : ndarray, (num_samples, n)
              NumPy array of input data.
    Returns
    -------
    jdata     : ndarray, (num_samples, num_samples-1, n)
              Jackknife sampling of data.
    """
    n = len(data)
    jdata = []
    for idx in range(n):
        jdata.append(np.delete(data, idx, axis=0))
    return np.array(jdata)


def jackknife_stat(data, stat_fn):
    """
    Computes the statistics over jackknife samples.
    See Chapter 11 of Ref [1].
    Params
    ------
    data      : ndarray, (num_samples, n)
              NumPy array of input data.
    stat_fn   : python function: data -> float or ndarray
              Function computing the desired statistics.
    Returns
    -------
    jstat     : ndarray, (num_samples, 1) or (num_samples, m)
              Statistics computed over jackknife samples.
    """
    jdata = jackknife_sample(data)
    return np.array([stat_fn(jd) for jd in jdata])


def jackknife_bias(data, stat_fn):
    """
    Computes the jackknife estimate of the bias of a statistic.
    See Chapter 11 of Ref [1].
    Params
    ------
    data      : ndarray, (num_samples, n)
              NumPy array of input data.
    stat_fn   : python function: data -> float or ndarray
              Function computing the desired statistics.
    Returns
    -------
    bias      : float
              Jackknife estimate of the bias.
    """
    n = len(data)
    orig_stat = stat_fn(data)
    jstat = jackknife_stat(data, stat_fn)
    jstat_mean = np.mean(jstat)
    return (n - 1) * (jstat_mean - orig_stat)


def jackknife_stderr(data, stat_fn):
    """
    Computes the jackknife estimate of the standard error.
    See Chapter 11 of Ref [1].
    Params
    ------
    data      : ndarray, (num_samples, n)
              NumPy array of input data.
    stat_fn   : python function: data -> float or ndarray
              Function computing the desired statistics.
    Returns
    -------
    stderr    : float
              Jackknife estimate of the standard error.
    """
    n = len(data)
    jstat = jackknife_stat(data, stat_fn)
    return (n - 1) ** 0.5 * np.std(jstat, ddof=0)


# ============================================================================
# Bootstrap: one-sample functions
# ============================================================================
# These functions are related to one-sample problems.


def bootstrap_sample(data, n_boots=1000, progbar=True):
    """
    Performs the bootstrap sampling of data.
    Params
    ------
    data      : ndarray, (n_samples, n)
              NumPy array of input data.
    n_boots   : int
              Number of bootstrap iterations (default: 1000).
    Returns
    -------
    bdata     : ndarray, (n_boots, n_samples, n)
              NumPy array of bootstrap samples.
    """
    data = np.array(data)
    n = data.shape[0]
    indeces = np.arange(n)
    bdata = []
    iterator = verb_iter(range(n_boots)) if progbar else range(n_boots)
    for it in iterator:
        bindeces = np.random.choice(indeces, size=n, replace=True)
        bdata.append(data[bindeces])
    return np.array(bdata)


def bootstrap_stat(data, stat_fn=np.mean, n_boots=1000, progbar=True):
    """
    Computes the bootstrap sampling of a statistics.
    Params
    ------
    data      : ndarray, (n_samples, n)
              NumPy array of input data.
    stat_fn   : Python function: data -> float or np.ndarray, (m,)
              Function computing the desired statistics (default: np.mean).
    n_boots   : int
              Number of bootstrap iterations (default: 1000).
    Returns
    -------
    bstats    : ndarray, (n_boots, 1) or ndarray, (n_boots, m)
              Bootstrap sampling of the statistics.
    """
    bdata = bootstrap_sample(data, n_boots=n_boots, progbar=progbar)
    bstats = np.array([stat_fn(bd) for bd in bdata])
    return bstats


def bootstrap_stderr(data, stat_fn=np.mean, n_boots=1000, progbar=True):
    """
    Computes the bootstrap estimate of the standard error of a statistics.
    See Algorithm 6.1 in Ref. [1].
    Params
    ------
    data      : ndarray, (n_samples, n)
              NumPy array of input data.
    stat_fn   : Python function: data -> float or np.ndarray, (m,)
              Function computing the desired statistics (default: np.mean).
    n_boots   : int
              Number of bootstrap iterations (default: 1000).
    Returns
    -------
    stderr_B  : float or np.ndarray, (m,)
              Bootstrap estimate of the standard error.
    """
    bstats = bootstrap_stat(data, stat_fn=stat_fn, n_boots=n_boots, progbar=progbar)
    return np.std(bstats, ddof=1, axis=0)


# ============================================================================
# Bootstrap: confidence interval functions
# ============================================================================
# Functions related to the estimation of confidence interval
# Note: only tested for 1D statistics, be careful if you think you can
# parallelize whatever you want.


def bootstrap_standard_ci(data, stat_fn, alpha=0.05, n_boots=1000, progbar=True):
    """
    Computes the standard confidence interval with coverage probability
    equal to (1 - 2 * alpha), using the bootstrap to estimate the standard
    error of the statistics.
    Params
    ------
    data      : ndarray, (n_samples, n)
              NumPy array of input data.
    stat_fn   : Python function: data -> float or np.ndarray, (m,)
              Function computing the desired statistics (default: np.mean).
    alpha     : float
              Alpha-th percentile point (default: 0.05).
    n_boots   : int
              Number of bootstrap iterations (default: 1000).
    Returns
    -------
    lo, up   : float or ndarray
              Confidence interval for the statistics
    """
    orig_stat = stat_fn(data)
    bstderr = bootstrap_stderr(data, stat_fn=stat_fn, n_boots=n_boots, progbar=True)
    z = sp_stats.norm.ppf(1 - alpha)
    return orig_stat - z * bstderr, orig_stat + z * bstderr


def bootstrap_studentt_ci(data, stat_fn, alpha=0.05, n_boots=1000, progbar=True):
    """
    Computes the studentized confidence interval with coverage probability
    equal to (1 - 2 * alpha), using the bootstrap to estimate the standard
    error of the statistics.
    Params
    ------
    data      : ndarray, (n_samples, n)
              NumPy array of input data.
    stat_fn   : Python function: data -> float or np.ndarray, (m,)
              Function computing the desired statistics (default: np.mean).
    alpha     : float
              Alpha-th percentile point (default: 0.05).
    n_boots   : int
              Number of bootstrap iterations (default: 1000).
    Returns
    -------
    lo, up   : float or ndarray
              Confidence interval for the statistics
    """
    df = np.array(data).shape[0] - 1
    orig_stat = stat_fn(data)
    bstderr = bootstrap_stderr(data, stat_fn=stat_fn, n_boots=n_boots, progbar=progbar)
    t = sp_stats.t.ppf(1 - alpha, df=df)
    return orig_stat - t * bstderr, orig_stat + t * bstderr


def bootstrap_t_ci(
    data, stat_fn, alpha=0.05, n_boots=1000, n_sub_boots=25, progbar=True
):
    """
    Computes the bootstrap-t confidence interval with coverage probability
    equal to (1 - 2*alpha).
    Params
    ------
    data        : ndarray, (n_samples, n)
                NumPy array of input data.
    stat_fn     : Python function: data -> float or np.ndarray, (m,)
                Function computing the desired statistics (default: np.mean).
    alpha       : float
                Alpha-th percentile point (default: 0.05).
    n_boots     : int
                Number of bootstrap iterations (default: 1000).
    n_sub_boots : int
                Number of bootstrap iterations to estimate the standard
                error of each bootstrap sample (default: 25).
    Returns
    -------
    lo, up   : float or ndarray
              Confidence interval for the statistics
    """
    orig_stat = stat_fn(data)
    bdata = bootstrap_sample(data, n_boots=n_boots, progbar=progbar)
    bstderr = np.std([stat_fn(bd) for bd in bdata], axis=0)
    tb_stat = []
    for bd in bdata:
        theta_b = stat_fn(bd)
        stderr_b = bootstrap_stderr(bd, stat_fn, n_boots=n_sub_boots, progbar=False)
        tb = (theta_b - orig_stat) / stderr_b
        tb_stat.append(tb)
    tb_stat = np.array(tb_stat)
    tb_up = np.quantile(tb_stat, q=alpha, axis=0)
    tb_lo = np.quantile(tb_stat, q=1 - alpha, axis=0)
    return orig_stat - tb_lo * bstderr, orig_stat - tb_up * bstderr


def bootstrap_percentile_ci(data, stat_fn, alpha=0.05, n_boots=1000, progbar=True):
    """
    Computes the bootstrap percentile confidence interval with coverage probability
    equal to (1 - 2*alpha).
    See Section 13.3 of Ref. [1].
    Params
    ------
    data        : ndarray, (n_samples, n)
                NumPy array of input data.
    stat_fn     : Python function: data -> float or np.ndarray, (m,)
                Function computing the desired statistics (default: np.mean).
    alpha       : float
                Alpha-th percentile point (default: 0.05).
    n_boots     : int
                Number of bootstrap iterations (default: 1000).
    n_sub_boots : int
                Number of bootstrap iterations to estimate the standard
                error of each bootstrap sample (default: 25).
    Returns
    -------
    lo, up   : float or ndarray
              Confidence interval for the statistics
    """
    bstat = bootstrap_stat(data, stat_fn, n_boots=n_boots, progbar=progbar)
    lo = np.quantile(bstat, q=alpha, axis=0)
    up = np.quantile(bstat, q=1 - alpha, axis=0)
    return lo, up


def bootstrap_bca_ci(data, stat_fn, alpha=0.05, n_boots=1000, progbar=True):
    """
    Computes the bootstrap BCa confidence interval with coverage probability
    equal to (1 - 2*alpha).
    See Section 14.3 of Ref. [1].
    Params
    ------
    data        : ndarray, (n_samples, n)
                NumPy array of input data.
    stat_fn     : Python function: data -> float or np.ndarray, (m,)
                Function computing the desired statistics (default: np.mean).
    alpha       : float
                Alpha-th percentile point (default: 0.05).
    n_boots     : int
                Number of bootstrap iterations (default: 1000).
    n_sub_boots : int
                Number of bootstrap iterations to estimate the standard
                error of each bootstrap sample (default: 25).
    Returns
    -------
    lo, up   : float or ndarray
              Confidence interval for the statistics
    """
    orig_stat = stat_fn(data)
    bstat = bootstrap_stat(data, stat_fn, n_boots=n_boots, progbar=progbar)
    z0 = sp_stats.norm.ppf(sum(bstat < orig_stat) / n_boots)
    theta = jackknife_stat(data, stat_fn=stat_fn)
    a = np.sum((np.mean(theta) - theta) ** 3) / (
        6 * np.sum((np.mean(theta) - theta) ** 2) ** 1.5
    )
    alpha1 = sp_stats.norm.cdf(
        z0 + (z0 + sp_stats.norm.ppf(alpha)) / (1 - a * (z0 + sp_stats.norm.ppf(alpha)))
    )
    alpha2 = sp_stats.norm.cdf(
        z0
        + (z0 + sp_stats.norm.ppf(1 - alpha))
        / (1 - a * (z0 + sp_stats.norm.ppf(1 - alpha)))
    )
    lo = np.quantile(bstat, q=alpha1, axis=0)
    up = np.quantile(bstat, q=alpha2, axis=0)
    return lo, up


# ============================================================================
# Bootstrap: hypothesis-testing functions
# ============================================================================
# Functions related to hypothesis testing with the bootstrap


def bootstrap_test(a, b, stat_fn, alternative="greater", n_boots=1000, progbar=True):
    """
    Computes the bootstrap test for a two-sample hypothesis testing problem.
    By default (alternative='grater') tests the null hypothesis that the two
    populations of a and b, say F and G, are identical, Hnull: F = G.
    See Algorithm 16.1 in Ref [1].
    Params
    ------
    a            : np.ndarray, (n,)
                 First class of observations
    b            : np.ndarray, (m,)
                 Second class of observations
    stat_fn      : python function: (a, b) -> float
                 Function computing the desired statistic
    alternative  : str
                 Can be 'two-sided', 'greater', or 'less'.
                 Call t* and t the bootstrap and original t statistic.
                 'two-sided' tests for t* being either bigger or smaller than t.
                 'greater' tests for t* being bigger than t.
                 'less' tests for t* being smaller than t.
    n_boots      : int
                 Number of bootstrap samples.
    Returns
    -------
    asl          : float
                 Achieved significance level (ASL).
    """
    if alternative not in ["two-sided", "greater", "less"]:
        raise RuntimeError("alternative can be 'two-sided', 'greater', or 'less'")
    x = np.concatenate((a, b))
    if x.ndim > 1:
        raise RuntimeError("a and b must be 1D NumPy arrays.")
    orig_tstat = stat_fn(a, b)
    indices = np.concatenate((np.zeros(len(a)), np.ones(len(b))))
    bindices = bootstrap_sample(indices, n_boots=n_boots, progbar=progbar)
    btstat = np.array([stat_fn(x[bi == 0], x[bi == 1]) for bi in bindices])
    if alternative == "two-sided":
        asl = sum(abs(btstat) > abs(orig_tstat)) / n_boots
    elif alternative == "greater":
        asl = sum(btstat > orig_tstat) / n_boots
    elif alternative == "less":
        asl = sum(btstat < orig_tstat) / n_boots
    return asl


def bootstrap_test_equalmeans(a, b, n_boots=1000, progbar=True):
    """
    Performs the bootstrap test for testing the equality of means.
    See Algorithm 16.2 in Ref [1].
    Params
    ------
    a         : np.ndarray, (m,)
              First class of observations
    b         : np.ndarray, (n,)
              Second class of observations
    n_boots   : int
              Number of bootstrap samples.
    Returns
    -------
    asl       : float
              Two-sided achieved significance level (ASL)
    """
    a = np.array(a)
    b = np.array(b)
    if a.ndim > 1 or b.ndim > 1:
        raise RuntimeError("a and b must be 1D NumPy arrays.")
    xbar = np.mean(np.concatenate((a, b)))
    atilde = a - np.mean(a) + xbar
    btilde = b - np.mean(b) + xbar

    def stat_fn(z, y):
        n = z.shape[0]
        m = y.shape[0]
        return (np.mean(z) - np.mean(y)) / (
            np.var(z, ddof=1) / n + np.var(y, ddof=1) / m
        ) ** 0.5

    orig_tstat = stat_fn(a, b)
    print(orig_tstat)
    btstat = bootstrap_n_stat(
        atilde, btilde, stat_fn=stat_fn, n_boots=n_boots, progbar=progbar
    )
    asl = sum(abs(btstat) > abs(orig_tstat)) / n_boots
    return asl


# ============================================================================
# Bootstrap: prediction error estimation
# ============================================================================
# Functions related to the use of the bootstrap to estimate the
# prediction error of an estimator


def bootstrap_632_error(x, y, model, loss, n_boots=1000, progbar=True):
    """
    Computes the bootstrap .632 estimate of the prediction error.
    See Chapter 17 of Ref [1].
    Params
    ------
    x         : np.ndarray, (n, m)
              Input data.
    y         : np.ndarray, (n,)
              Target data.
    model     : python class
              Class representing the model.
              Should have a method `fit(self, x, y)` to fit the model,
              returning `self`, and a method `predict(self, x)` to
              perform the prediction, returning a NumPy array.
    loss      : python function: (x, y) -> float
              Loss function.
    n_boots   : int
              Number of bootstrap samples.
    Returns
    -------
    b632_err  : float
              Bootstrap .632 estimate of the prediction error.
    """
    x = np.array(x)
    y = np.array(y)
    if y.ndim > 1:
        raise RuntimeError("y must be a 1D NumPy array.")
    n = x.shape[0]
    indices = np.arange(n)
    counts = np.zeros(y.shape)
    eps0 = np.zeros(y.shape)
    app_err = loss(y, model.fit(x, y).predict(x))
    iterator = verb_iter(range(n_boots)) if progbar else range(n_boots)
    for it in iterator:
        tr_bindices = np.random.choice(indices, size=n, replace=True)
        te_bindices = np.setdiff1d(indices, tr_bindices)
        bpred = model.fit(x[tr_bindices], y[tr_bindices]).predict(x[te_bindices])
        counts[te_bindices] += 1
        eps0[te_bindices] += loss(y[te_bindices], bpred)
    eps0 = np.mean(eps0 / counts)
    return 0.368 * app_err + 0.632 * eps0


# ============================================================================
# Bootstrap: parametric bootstrap functions
# ============================================================================
# Functions related to the parametric bootstrap
# Note: only tested in the 1D case.


def bootstrap_parametric_sample(distrib_fn, size, n_boots=1000, progbar=True):
    """
    Computes the bootstrap sampling from an assumed parametric
    distribution of data.
    See Section 6.5 in Ref. [1].
    Params
    ------
    distrib_fn  : python function
                Distribution function. Only called with its keyword
                'size' to specify the size of each bootstrap sample.
    size        : int
                Size of each bootstrap sample
    n_boots     : int
                Number of bootstrap iterations (default: 1000)
    Returns
    -------
    bdata       : np.ndarray, (n_boots, n_samples, n)
    """
    bdata = []
    iterator = ver_iter(range(n_boots)) if progbar else range(n_boots)
    for it in iterator:
        bdata.append(distrib_fn(size=size))
    return np.array(bdata)


def bootstrap_parametric_stat(distrib_fn, size, stat_fn, n_boots=1000, progbar=True):
    """
    Computes the bootstrap sampling of a statistics from an assumed
    parametric distribution of data.
    See Section 6.5 in Ref. [1].
    Params
    ------
    distrib_fn  : python function
                Distribution function. Only called with its keyword
                'size' to specify the size of each bootstrap sample.
    size        : int
                Size of each bootstrap sample
    stat_fn     : Python function: data -> float or np.ndarray, (m,)
                Function computing the desired statistics (default: np.mean).
    n_boots     : int
                Number of bootstrap iterations (default: 1000)
    Returns
    -------
    bstats      : np.ndarray, (n_boots, 1) or np.ndarray, (n_boots, m)
    """
    bdata = bootstrap_parametric_sample(
        distrib_fn, size=size, n_boots=n_boots, progbar=progbar
    )
    bstats = np.array([stat_fn(bd) for bd in bdata])
    return bstats


def bootstrap_parametric_stderr(distrib_fn, size, stat_fn, n_boots=1000, progbar=True):
    """
    Computes the bootstrap estimate of the standard error of a statistics
    from an assumed parametric distribution of data.
    See Section 6.5 in Ref. [1].
    Params
    ------
    distrib_fn  : python function
                Distribution function. Only called with its keyword
                'size' to specify the size of each bootstrap sample.
    size        : int
                Size of each bootstrap sample
    stat_fn     : Python function: data -> float or np.ndarray, (m,)
                Function computing the desired statistics (default: np.mean).
    n_boots     : int
                Number of bootstrap iterations (default: 1000)
    Returns
    -------
    stderr_B   : float or np.ndarray, (m,)
               Bootstrap estimate of the standard error.
    """
    bstats = bootstrap_parametric_stat(
        distrib_fn,
        size=size,
        stat_fn=stat_fn,
        n_boots=n_boots,
        progbar=progbar,
    )
    return np.std(bstats, ddof=1, axis=0)


# ============================================================================
# Bootstrap: n-sample related functions
# ============================================================================
# Functions related to the application of the bootstrap to an arbitrary
# number of samples.


def bootstrap_n_sample(*data, n_boots=1000, progbar=True):
    """
    Performs the bootstrap sampling of an arbitrary number of data.
    Useful e.g. in a two-sample problem.
    See Chapter 8 in Ref. [1].
    Params
    ------
    data      : ndarrays, (n_samples, n)
              NumPy arrays of input data.
    n_boots   : int
              Number of bootstrap iterations (default: 1000).
    Returns
    -------
    bdata     : list of ndarray, (n_boots, n_samples, n)
              NumPy arrays of bootstrap samples
    """
    bdata = []
    for subdata in data:
        sub_bdata = bootstrap_sample(subdata, n_boots=n_boots, progbar=progbar)
        bdata.append(sub_bdata)
    return bdata


def bootstrap_n_stat(*data, stat_fn=None, n_boots=1000, progbar=True):
    """
    Performs the bootstrap sampling of a statistics on an arbitrary number of data.
    Useful e.g. in a two-sample problem.
    See Chapter 8 in Ref. [1].
    Params
    ------
    data      : ndarrays, (n_samples, n)
              NumPy arrays of input data.
    stat_fn   : python function: [*data] -> float or np.ndarray, (m,)
              Function computing the desired statistics (default: None).
              Note that a stat_fn has to be specified.
              It should accept an arbitrary number of data items, and compute
              the statistic of interest.
              The order of data items fed into stat_fn is the same as the
              order in which *data is provided.
    n_boots   : int
              Number of bootstrap iterations (default: 1000).
    Returns
    -------
    bstats    : list of ndarray, (n_boots, 1) or ndarray, (n_boots, m)
              NumPy arrays of bootstrap samples
    """
    if stat_fn is None:
        raise RuntimeError("stat_fn not specified.")
    bdata = bootstrap_n_sample(*data, n_boots=n_boots, progbar=progbar)
    bstats = np.array([stat_fn(*sub_data) for sub_data in zip(*bdata)])
    return bstats


def bootstrap_n_stderr(*data, stat_fn=None, n_boots=1000, progbar=True):
    """
    Performs the bootstrap estimate of a statistics on an arbitrary number of data items.
    Useful e.g. in a two-sample problem.
    See Chapter 8 in Ref. [1].
    Params
    ------
    data      : ndarrays, (n_samples, n)
              NumPy arrays of input data.
    stat_fn   : python function: [*data] -> float or np.ndarray, (m,)
              Function computing the desired statistics (default: None).
              Note that a stat_fn has to be specified.
              It should accept an arbitrary number of data items, and compute
              the statistic of interest.
              The order of data items fed into stat_fn is the same as the
              order in which *data is provided.
    n_boots   : int
              Number of bootstrap iterations (default: 1000).
    Returns
    -------
    stderr_B  : float or np.ndarray, (m,)
              Bootstrap estimate of the standard error of the statistics.
    """
    bstats = bootstrap_n_stat(*data, stat_fn=stat_fn, n_boots=n_boots, progbar=progbar)
    return np.std(bstats, ddof=1, axis=0)


# ============================================================================
# Bootstrap: time-series or correlated data
# ============================================================================
# Functions realated to the application of the bootstrap to correlated
# or time-series data.


def bootstrap_moving_block(data, block, n_boots=1000, progbar=True):
    """
    Performs the moving-block bootstrap for bootstrapping time-series data.
    See Section 8.6 in Ref. [1].
    Params
    ------
    data      : ndarray, (n_samples, n)
              NumPy array of input data.
    block     : int
              Length of the moving block.
    n_boots   : int
              Number of bootstrap iterations (default: 1000).
    Returns
    -------
    bdata     : ndarray, (n_boots, n_samples, n)
              NumPy array of bootstrap samples.
    """
    data = np.array(data)
    n = data.shape[0]
    k = int(n / block)
    blocked_data = np.array([data[i : i + block] for i in range(n - block + 1)])
    m = blocked_data.shape[0]
    indeces = np.arange(m)
    bdata = []
    iterator = verb_iter(range(n_boots)) if progbar else range(n_boots)
    for it in iterator:
        bindeces = np.random.choice(indeces, size=k, replace=True)
        bdata.append(np.concatenate(blocked_data[bindeces]))
    return np.array(bdata)
