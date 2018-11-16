import numpy as np
from keras.layers import Input, Dense, Concatenate
from keras.engine.topology import Network
from keras.optimizers import Adam, RMSprop
from keras.models import Model
import tensorflow as tf


__all__ = ['GARMConfig',
           'GARM']


class GARMConfig(object):
    def __init__(self):
        self.n_rep = 2
        self.n_reg = 2
        self.n_dis = 2
        self.dim_rep = 8
        self.dim_reg = 8
        self.dim_dis = 8

        self.n_input = 25


class GARM(object):
    def __init__(self,
                 config: GARMConfig):
        self._config = config

        self._rep_net = self.build_representation()
        self._reg_net = self.build_regression()

        self._dis_net = self.build_discriminator()
        self._dis_net.compile(RMSprop(0.001),
                              loss=['binary_crossentropy'])  # 训练判别器的，输入为x, t, y

        x_input = Input(shape=(self._config.n_input, ))
        t_input = Input(shape=(1, ))
        x_rep = self._rep_net(x_input)
        y_pred = self._reg_net([x_rep, t_input])
        self._generator = Model(inputs=(x_input, t_input), outputs=y_pred)  # 训练生成器，输入为x, t
        self._generator.compile(RMSprop(0.001),
                                loss=['mean_squared_error'])

        self._dis_net_fixed = Network(inputs=self._dis_net.input, outputs=self._dis_net.output)
        self._dis_net_fixed.trainable = False
        logits = self._dis_net_fixed([x_rep, t_input, y_pred])
        self._gan = Model(inputs=(x_input, t_input), outputs=logits)  # 训练生成器用的，只需要x和t的输入
        self._gan.compile(RMSprop(0.001),
                          loss=['binary_crossentropy'])

    def build_representation(self):
        inputs = Input(shape=(self._config.n_input, ))
        x = inputs
        for _ in range(self._config.n_rep):
            x = Dense(self._config.dim_rep, activation='relu')(x)
        model = Model(inputs=inputs, outputs=x)
        return model

    def build_regression(self):
        if self._config.n_rep == 0:
            input_dim = self._config.n_input
        else:
            input_dim = self._config.dim_rep
        inputs = Input(shape=(input_dim, ))
        t = Input(shape=(1, ))
        x = Concatenate()([inputs, t])
        for _ in range(self._config.n_reg):
            x = Dense(self._config.dim_reg, activation='relu')(x)
        x = Dense(1, activation=None)(x)
        model = Model(inputs=(inputs, t), outputs=x)
        return model

    def build_discriminator(self):
        if self._config.n_rep == 0:
            input_dim = self._config.n_input
        else:
            input_dim = self._config.dim_rep
        inputs = Input(shape=(input_dim, ))
        t = Input(shape=(1, ))
        y = Input(shape=(1, ))
        x = Concatenate()([inputs, t, y])
        for _ in range(self._config.n_dis):
            x = Dense(self._config.dim_dis, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=(inputs, t, y), outputs=x)
        return model

    def train(self, train_set, val_set, batch_size=128, iterations=1000):
        x, t, y = train_set.x, train_set.t, train_set.y

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        num = x.shape[0]

        # --------------------------------
        # supervised training of predictor
        # --------------------------------
        # for i in range(iterations):
        #     idx = np.random.randint(0, x.shape[0], batch_size)
        #     train_x, train_t, train_y = x[idx], t[idx], y[idx]
        #
        #     self._generator.train_on_batch((train_x, train_t), train_y)
        self._generator.fit([x, t], y,
                            batch_size=128,
                            epochs=int(iterations * batch_size / num),
                            validation_data=([val_set.x, val_set.t], val_set.y))

        # -----------------
        # adversarial train
        # -----------------
        for i in range(iterations):
            idx = np.random.randint(0, num, batch_size)
            train_x, train_t, train_y = x[idx], t[idx], y[idx]

            fake_y = self._generator.predict([train_x, 1-train_t])
            x_rep = self._rep_net.predict(train_x)
            self._dis_net.train_on_batch([x_rep, train_t, train_y], valid)
            self._dis_net.train_on_batch([x_rep, 1-train_t, fake_y], fake)

            self._gan.train_on_batch([train_x, 1-train_t], valid)

    def predict(self, x, t, batch_size=128):
        y = self._generator.predict([x, t], batch_size=batch_size)
        return y

    def evaluate(self):
        pass
