from keras.layers import Input, Dense, Dropout, BatchNormalization, Add
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
import keras.backend as K


def build_optimizer(optimizer, learning_rate):
    if optimizer == 'adam':
        opt = Adam(learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate)
    else:
        opt = SGD(learning_rate)
    return opt


def custom_loss(loss, n_class):
    if loss == 'crossentropy':
        if n_class <= 2:
            return 'binary_crossentropy'
        else:
            return 'categorical_crossentropy'
    elif loss == 'focal_loss':
        if n_class <= 2:
            return focal_loss
    # elif loss == 'hinge':
    #     if n_class == 2:
    #         return 'hinge'
    #     else:
    #         return 'categorical_hinge'


def focal_loss(y_true, y_pred):
    loss0 = - (1 - y_true) * y_pred ** 2 * K.log(1 - y_pred + 1e-8)
    loss1 = - y_true * (1 - y_pred) ** 2 * K.log(y_pred + 1e-8)
    return loss0 + loss1


callbacks = [
    EarlyStopping(monitor='val_acc', min_delta=0.0005, patience=5)
]


param_dist = {
    'layers': [3, 5, 10],
    'units': [25, 12, 8, 4],
    'activation_fn': ['relu', 'sigmoid', 'elu'],
    'use_dropout': [0],
    'drop_rate': [0.1, 0.2, 0.3, 0.5],
    'use_bn': [1],
    'optimizer': ['adam', 'rmsprop', 'sgd'],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'loss_fn': ['crossentropy'],
    'regularizer': [l1, l2],
    'regular_coef': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
}

sample_param = {
    'layers': 1,
    'units': 12,
    'activation_fn': 'relu',
    'use_dropout': 0,
    'drop_rate': 0,
    'use_bn': 1,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss_fn': 'crossentropy',
    'regularizer': l2,
    'regular_coef': 0.01
}

fit_params = {
    'epochs': 1000,
    'batch_size': 128,
    'verbose': 0,
    'validation_split': 0.2,
    'callbacks': callbacks
}


class MyModel(object):
    def __init__(self,
                 n_input,
                 n_classes):
        self.n_input = n_input
        self.n_classes = n_classes

    def __call__(self,
                 layers,
                 units,
                 activation_fn,
                 use_dropout,
                 drop_rate,
                 use_bn,
                 optimizer,
                 learning_rate,
                 loss_fn,
                 regularizer,
                 regular_coef):
        K.clear_session()

        inputs = Input(shape=(self.n_input, ))
        x = inputs

        use_bias = False if use_bn else True

        for _ in range(layers):
            x = Dense(units,
                      activation=activation_fn,
                      use_bias=use_bias,
                      kernel_regularizer=regularizer(regular_coef))(x)
            if use_bn:  # preferred to use batch normalization
                x = BatchNormalization()(x)
            elif use_dropout and drop_rate != 0:
                x = Dropout(drop_rate)(x)

        if self.n_classes <= 2:
            outputs = Dense(1,
                            activation='sigmoid',
                            kernel_regularizer=regularizer(regular_coef))(x)
        else:
            outputs = Dense(self.n_classes,
                            activation='softmax',
                            kernel_regularizer=regularizer(regular_coef))(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(build_optimizer(optimizer, learning_rate),
                      loss=custom_loss(loss_fn, self.n_classes),
                      metrics=['acc'],
                      callbacks=callbacks)

        return model


residual_param_dist = {
    'layers': [1, 2, 3, 4, 5],
    'units': [25, 12, 8],
    'activation_fn': ['elu', 'sigmoid', 'relu'],
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'learning_rate': [0.01, 0.001, 0.0001],
    'dropout_rate': [0.1, 0.2, 0.3, 0.5],
    'loss_fn': ['crossentropy'],
    'regularizer': [l1, l2],
    'regular_coef': [0.1, 0.01, 0.001, 0.0001]
}


class ResidualModel(object):
    def __init__(self, n_input, n_class):
        self.n_input = n_input
        self.n_class = n_class

    def __call__(self,
                 layers,
                 units,
                 activation_fn,
                 optimizer,
                 learning_rate,
                 dropout_rate,
                 loss_fn,
                 regularizer,
                 regular_coef):
        K.clear_session()

        inputs = Input(shape=(self.n_input, ))
        x = Dense(units,
                  activation=activation_fn,
                  kernel_regularizer=regularizer(regular_coef))(inputs)

        for _ in range(layers):
            i = x
            x = Dense(units,
                      activation=activation_fn,
                      kernel_regularizer=regularizer(regular_coef))(x)
            x = Dense(units,
                      kernel_regularizer=regularizer(regular_coef))(x)
            x = Dropout(dropout_rate)(x)
            x = Add()([i, x])
            x = BatchNormalization()(x)

        if self.n_class <= 2:
            outputs = Dense(1,
                            activation='sigmoid',
                            kernel_regularizer=regularizer(regular_coef))(x)
        else:
            outputs = Dense(self.n_class,
                            activation='softmax',
                            kernel_regularizer=regularizer(regular_coef))(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(build_optimizer(optimizer, learning_rate),
                      loss=custom_loss(loss_fn, self.n_class),
                      metrics=['acc'])
        return model

