import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from .models import MyModel, param_dist, fit_params, sample_param, ResidualModel, residual_param_dist
from data import HF, MyScaler


def run_experiments(n_iter=1000):
    data = HF.load()
    x = np.concatenate((data.x, data.t), -1)
    scaler = MyScaler()
    x = scaler.fit_transform(x)

    n_input = x.shape[1]
    if data.y.ndim == 1:
        n_class = 2
    else:
        n_class = data.y.shape[1]

    # model = MyModel(n_input, n_class)
    model = ResidualModel(n_input, n_class)
    wrapper_classifier = KerasClassifier(model)

    rsc = RandomizedSearchCV(wrapper_classifier,
                             # param_distributions=param_dist,
                             param_distributions=residual_param_dist,
                             n_iter=n_iter,
                             n_jobs=4,
                             cv=5,
                             return_train_score=True)

    rsc.fit(x, data.y, **fit_params)

    print(rsc.best_score_)
    cv_scores = pd.DataFrame(rsc.cv_results_)
    cv_scores.to_csv('./result/cv_scores.csv', mode='a', index=False)


def test_model():
    data = HF.load()
    x = np.concatenate((data.x, data.t), -1)
    # 归一化
    scaler = MyScaler()
    x = scaler.fit_transform(x)

    n_input = x.shape[1]
    if data.y.ndim == 1:
        n_class = 2
    else:
        n_class = data.y.shape[1]

    model = MyModel(n_input, n_class)(**sample_param)
    model.summary()

    model.fit(x, data.y, **fit_params)
