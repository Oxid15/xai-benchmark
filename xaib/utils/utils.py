from typing import List

import numpy as np


def rmse(x, y):
    err = x - y
    mse = np.sum(err * err)
    return np.sqrt(mse)


def batch_rmse(bx, by):
    return [rmse(x, y) for (x, y) in zip(bx, by)]


def entropy(x):
    x += 1e-6
    return -np.sum(x * np.log2(x))


def gini(x):
    n = len(x)
    ad = 0
    for i in range(n):
        for j in range(n):
            ad += np.abs(x[i] - x[j])
    if np.mean(x) == 0:
        return 0
    else:
        gini = ad / (2 * n * n * np.mean(x))
    return gini


def batch_gini(x):
    return [gini(i) for i in x]


def batch_count_eq(x, y) -> List[bool]:
    return [all(xi == yi) for xi, yi in zip(x, y)]


def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)

    # If all values are 0 then return original
    # If all values are equal, then return ones
    if min_val == max_val:
        if min_val == 0:
            return x
        else:
            return np.ones_like(x)

    x_scaled = (x + np.abs(min_val)) / (max_val + np.abs(min_val))
    return x_scaled
