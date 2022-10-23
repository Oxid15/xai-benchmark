import numpy as np


def rmse(x, y):
    err = x - y
    mse = np.sum(err * err)
    return np.sqrt(mse)
