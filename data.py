import os

import pandas as pd
import numpy as np


def merge_data(data_path='D:/data/ihdp/ucgs_a_10712097_sm0001/code/gen/YB_overlap',
               out_file='./resources/ihdpB_overlap.npz'):
    files = os.listdir(data_path)
    result = {str(i): None for i in range(len(files))}
    for i, file in enumerate(files):
        file_path = os.path.join(data_path, file)
        d = pd.read_csv(file_path)
        result[str(i)] = d.values

    np.savez_compressed(out_file, **result)


class DataSet(object):
    pass


class IHDP(DataSet):
    def __init__(self, x, t, y, y1, y0):
        self.x = x
        self.t = t
        self.y = y
        self.y1 = y1
        self.y0 = y0

    @classmethod
    def load(cls, i):
        x_path = './resources/x.npy'
        y_path = './resources/ihdpB_overlap.npz'
        t_path = './resources/t.npy'
        x = np.load(x_path)
        y = np.load(y_path)[str(i)]
        yf = np.reshape(y[:, 0], (-1, 1))
        y0 = y[:, 1]
        y1 = y[:, 2]
        t = np.reshape(np.load(t_path), -1)
        t = pd.get_dummies(t).values
        return cls(x, t, yf, y1, y0)

    def split(self, train=0.56, val=0.24):
        num = self.x.shape[0]
        train_num, val_num = int(num * train), int(num * val)
        index = np.arange(num)
        np.random.shuffle(index)
        train_idx = index[0:train_num]
        val_idx = index[train_num: train_num+val_num]
        test_idx = index[train_num+val_num:]
        train_set, val_set, test_set = self.subset(train_idx), self.subset(val_idx), self.subset(test_idx)
        return train_set, val_set, test_set

    def subset(self, index):
        return IHDP(self.x[index], self.t[index], self.y[index], self.y1[index], self.y0[index])


if __name__ == "__main__":
    merge_data()
