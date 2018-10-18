import numpy as np

from .model import *
from data import IHDP
from metric import sqrt_pehe


def run_experiments(num_experiments=1000):
    config = GARMConfig()
    model = GARM(config)

    for i in range(num_experiments):
        data_set = IHDP.load(i)
        data_set.t = data_set.t[:, 1].reshape(-1, 1)
        train_set, val_set, test_set = data_set.split()
        model.fit(train_set)

        pred_ite = model.ite(test_set)
        true_ite = np.transpose(np.vstack((test_set.y0, test_set.y1)))
        pehe = sqrt_pehe(true_ite, pred_ite)
        print(pehe)
