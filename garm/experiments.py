import numpy as np

# from .model import *
from .models import *
from data import IHDP, HF, MyScaler
from metric import sqrt_pehe


def run_experiments(num_experiments=1000, data_set_type='IHDP'):
    config = GARMConfig()
    if data_set_type == 'HF':
        config.categorical = 1
        config.n_input = 105
        config.n_treat = 3

    acc = []
    for i in range(num_experiments):
        if data_set_type == 'HF':
            data_set = HF.load()
        else:
            data_set = IHDP.load(i)

        train_set, val_set, test_set = data_set.split()
        scaler = MyScaler()
        train_set.x = scaler.fit_transform(train_set.x)
        val_set.x = scaler.transform(val_set.x)
        test_set.x = scaler.transform(test_set.x)

        model = GARM(config)
        model.train(train_set, val_set, ad_train=True)

        metrics = model.evaluate(test_set)
        for n, m in zip(model.metrics_names, metrics):
            print('{}: {}'.format(n, m))
        acc.append(metrics[1])

        # y1 = model.predict(test_set.x, np.ones_like(test_set.t))
        # y0 = model.predict(test_set.x, np.zeros_like(test_set.t))
        # pred_ite = np.concatenate((y0, y1), -1)
        # true_ite = np.transpose(np.vstack((test_set.y0, test_set.y1)))
        # pehe = sqrt_pehe(true_ite, pred_ite)
        # print('pehe of {}th experiments is {}'.format(i, pehe))

    acc_mean = np.mean(acc)
    acc_std = np.std(acc)
    print("mean of acc is {}".format(acc_mean))
    print('std of acc is {}'.format(acc_std))

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
