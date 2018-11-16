import numpy as np

# from .model import *
from .models import *
from data import IHDP
from metric import sqrt_pehe


def run_experiments(num_experiments=1000):
    config = GARMConfig()

    for i in range(num_experiments):
        data_set = IHDP.load(i)
        train_set, val_set, test_set = data_set.split()

        model = GARM(config)
        model.train(train_set, val_set)

        y1 = model.predict(test_set.x, np.ones_like(test_set.t))
        y0 = model.predict(test_set.x, np.zeros_like(test_set.t))
        pred_ite = np.concatenate((y0, y1), -1)
        true_ite = np.transpose(np.vstack((test_set.y0, test_set.y1)))
        pehe = sqrt_pehe(true_ite, pred_ite)
        print('pehe of {}th experiments is {}'.format(i, pehe))

# def run_experiments(num_experiments=1000):
#     config = GARMConfig()
#     model = GARM(config)
#     pehes = []
#
#     for i in range(num_experiments):
#         print('----------------------------')
#         print('run {} of {} experiments'.format(i, num_experiments))
#
#         data_set = IHDP.load(i)
#         data_set.t = data_set.t[:, 1].reshape(-1, 1)
#         train_set, val_set, test_set = data_set.split()
#         model.fit(train_set)
#
#         pred_ite = model.ite(test_set)
#         true_ite = np.transpose(np.vstack((test_set.y0, test_set.y1)))
#         pehe = sqrt_pehe(true_ite, pred_ite)
#         print('pehe of {}th experiments is {}'.format(i, pehe))
#         pehes.append(pehe)
#         print('----------------------------')
#
#     print('meas of pehe is {}'.format(np.mean(pehes)))
#     print('std of pehe is {}'.format(np.std(pehes)))
