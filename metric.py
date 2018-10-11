import numpy as np


def pehe(true_ite, pred_ite):
    assert pred_ite.shape[1] == 2
    if isinstance(true_ite, int):
        diff = true_ite - (pred_ite[:, 1] - pred_ite[:, 0])
    else:
        diff = (true_ite[:, 1] - true_ite[:, 0]) - (pred_ite[:, 1] - pred_ite[:, 0])
    squared = diff ** 2
    mean = np.mean(squared)
    return mean


def sqrt_pehe(true_ite, pred_ite):
    return np.sqrt(pehe(true_ite, pred_ite))


def mse(true_y, pred_y):
    return np.mean(((true_y - pred_y) ** 2))
