import numpy as np


def pehe(true_ite, pred_ite):
    assert true_ite.shape == pred_ite
    assert true_ite.shape[1] == 2

    return np.mean((true_ite[:, 1] - true_ite[:, 0]) - (pred_ite[:, 1] - pred_ite[:, 0]) ** 2)


def mse(true_y, pred_y):
    return np.mean(((true_y - pred_y) ** 2))
