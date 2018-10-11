import numpy as np

from .model import *
from data import IHDP
from metric import sqrt_pehe


def run_experiments(n_experiments=1000):
    gan_ite_config = GANITEConfig()
    cf_block = CFBlock(gan_ite_config)

    ite_block = ITEBlock(gan_ite_config)

    for i in range(1, n_experiments+1):
        data_set = IHDP.load(i)
        train_set, val_set, test_set = data_set.split()
        cf_block.fit(train_set)
        y_bar = cf_block.gen_y_bar(train_set)

        class CompleteSet(object):
            pass

        complete_set = CompleteSet()
        complete_set.x = train_set.x
        complete_set.y = y_bar

        ite_block.fit(complete_set)
        y_hat = ite_block.gen_y_hat(test_set)
        true_ite = np.transpose(np.vstack((test_set.y0, test_set.y1)))
        pehe = sqrt_pehe(true_ite, y_hat)
        print(pehe)
