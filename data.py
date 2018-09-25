import os

import pandas as pd
import numpy as np


def merge_data(data_path='D:/data/ihdp/ucgs_a_10712097_sm0001/code/gen/YB',
               out_file='./resources/ihdpB.npz'):
    files = os.listdir(data_path)
    result = {str(i): None for i in range(len(files))}
    for i, file in enumerate(files):
        file_path = os.path.join(data_path, file)
        d = pd.read_csv(file_path)
        result[str(i)] = d.values

    np.savez_compressed(out_file, **result)


if __name__ == "__main__":
    merge_data()
