import json

import tensorflow as tf

from models import BaseModel


__all__ = ['GARM',
           "GARMConfig"]


class GARMConfig(object):
    def __init__(self):
        self.input_dim = 25

        self.hidden_units = 8
        self.hidden_layers = 5

        self.epochs = 10000
        self.learning_rate = 0.001
        self.batch_size = 64

        self.save_n_times = 1
        self.save_path = './model_save/garm'

    @property
    def buffer_size(self):
        return self.batch_size * 10

    @property
    def save_n_epochs(self):
        return int(self.epochs / self.save_n_times)

    def save(self, file):
        with open(file, 'w', encoding='utf-8') as wf:
            json.dump(self.__dict__, wf, indent=2, separators=(',', ': '), default=lambda o: o.__dict__)

    @classmethod
    def load(cls, file):
        with open(file, 'r', encoding='utf-8') as rf:
            con = json.load(rf)
        c = cls()
        for k, v in con.items():
            setattr(c, k, v)
        return c

    def __eq__(self, other):
        if isinstance(other, GARMConfig):
            if len(self.__dict__) != len(other.__dict__):
                return False
            for f, s in zip(self.__dict__.items(), other.__dict__.items()):
                if f != s:
                    return False
        return NotImplementedError


class GARM(BaseModel):
    def __init__(self,
                 config: GARMConfig,
                 name='garm'):
        super().__init__(config, name)

    def _build_graph(self):
        with tf.variable_scope(self._name):
            self._placeholder_def()
            self._set_def()
            self._main_graph()
            self._loss_def()
            self._train_def()

    def _placeholder_def(self):
        with tf.name_scope("input"):
            self._input_x = tf.placeholder(tf.float32, [None, self._config.input_dim], name='input_x')
            self._input_t = tf.placeholder(tf.float32, [None, 1], name='input_t')
            self._input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')

    def _set_def(self):
        with tf.name_scope('data_set'):
            self._data_set = tf.data.Dataset.from_tensor_slices((self._input_x, self._input_t, self._input_y))
            self._data_set = self._data_set.repeat().shuffle(self._config.buffer_size).batch(self._config.batch_size)

            self._iter = self._data_set.make_initializable_iterator()
            self._x_batch, self._t_batch, self._y_batch = self._iter.get_next()

    def _main_graph(self):
        def gen(x, t, reuse=False):
            with tf.variable_scope('generator', reuse=reuse):
                i = tf.concat((x, t), -1)
                for _ in range(self._config.hidden_layers - 1):
                    i = tf.contrib.layers.fully_connected(i, self._config.hidden_units)
                y_hat = tf.contrib.layers.fully_connected(i, 1, activation_fn=None)
                return y_hat

        def dis(x, t, y, reuse=False):
            with tf.variable_scope('discriminator', reuse=reuse):
                i = tf.concat((x, t, y), -1)
                for _ in range(self._config.hidden_layers - 1):
                    i = tf.contrib.layers.fully_connected(i, self._config.hidden_units)
                logits = tf.contrib.layers.fully_connected(i, 1, activation_fn=None)
                return logits

        self._y_hat = gen(self._x_batch, self._t_batch)
        self._fake_y_hat = gen(self._x_batch, 1 - self._t_batch, reuse=True)

        self._logits = dis(self._x_batch, self._t_batch, self._y_batch)
        self._fake_logits = dis(self._x_batch, self._t_batch, self._fake_y_hat, reuse=True)

        batch_size = tf.shape(self._x_batch)[0]
        y0 = gen(self._x_batch, tf.zeros((batch_size, 1)), reuse=True)
        y1 = gen(self._x_batch, tf.ones((batch_size, 1)), reuse=True)
        self._ite = tf.concat((y0, y1), -1)

    def _loss_def(self):
        with tf.name_scope('losses'):
            with tf.name_scope('supervised_loss'):
                self._supervised_loss = tf.losses.mean_squared_error(self._y_batch,
                                                                     self._y_hat)

            with tf.name_scope('gan_loss'):
                self._g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self._fake_logits),
                                                               self._fake_logits)

                self._d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self._logits), self._logits) + \
                    tf.losses.sigmoid_cross_entropy(tf.zeros_like(self._fake_logits), self._fake_logits)

            with tf.name_scope('reward'):
                self._reward = tf.exp(tf.negative(self._supervised_loss)) + self._fake_logits
                self._reward = tf.negative(self._reward)

    def _train_def(self):
        # supervised
        self._train_p = tf.train.AdamOptimizer().minimize(self._supervised_loss)

        # train d
        g_vars = tf.trainable_variables(self._name + '/generator')
        d_vars = tf.trainable_variables(self._name + "/discriminator")
        self._train_d = tf.train.AdamOptimizer().minimize(self._d_loss, var_list=d_vars)

        # maximize reward
        self._train_h = tf.train.AdamOptimizer().minimize(self._reward, var_list=g_vars)

    def loss(self, data_set):
        y_f_loss = self._sess.run(self._supervised_loss, feed_dict={self._x_batch: data_set.x,
                                                                    self._t_batch: data_set.t,
                                                                    self._y_batch: data_set.y})
        return y_f_loss

    def fit(self, data_set):
        self._sess.run(self._init)
        self._sess.run(self._iter.initializer, feed_dict={self._input_x: data_set.x,
                                                          self._input_t: data_set.t,
                                                          self._input_y: data_set.y})

        for i in range(self._config.epochs):
            self._sess.run(self._train_p)

            if i % 100 == 0 or i == self._config.epochs-1:
                y_f_loss = self.loss(data_set)
                print('loss of epoch {} is {}'.format(i, y_f_loss))

        for _ in range(self._config.epochs):
            self._sess.run(self._train_d)

        for _ in range(self._config.epochs):
            self._sess.run(self._train_d)
            self._sess.run(self._train_h)

    def ite(self, data_set):
        return self._sess.run(self._ite, feed_dict={self._x_batch: data_set.x})
