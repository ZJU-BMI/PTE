import os

import tensorflow as tf

from models import BaseModel


__all__ = ['GANITEConfig',
           'CFBlock',
           'ITEBlock']


class GANITEConfig(object):
    def __init__(self):
        self.k = 2
        self.input_dim = 25

        self.hidden_units = 8
        self.hidden_layers = 5

        self.alpha = 2
        self.beta = 5

        self.epochs = 1000
        self.learning_rate = 0.001
        self.batch_size = 64

        self.save_n_times = 1
        self.cf_save_path = './model_save/ganite/cfblock'
        self.ite_save_path = './model_save/ganite/iteblock'

    @property
    def buffer_size(self):
        return self.batch_size * 10

    @property
    def save_n_epochs(self):
        return int(self.epochs / self.save_n_times)


class CFBlock(BaseModel):
    def __init__(self,
                 config: GANITEConfig,
                 name='cf'):
        super().__init__(config, name)

    def _build_graph(self):
        with tf.variable_scope(self._name):
            self._placeholder_def()
            self._set_def()
            self._gan_def()
            self._loss_def()
            self._train_def()

    def _placeholder_def(self):
        with tf.name_scope('input'):
            self._input_x = tf.placeholder(tf.float32, [None, self._config.input_dim], name="input_x")
            self._input_t = tf.placeholder(tf.float32, [None, self._config.k], name="input_x")
            self._input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')

    def _set_def(self):
        with tf.name_scope('data_set'):
            self._data_set = tf.data.Dataset.from_tensor_slices((self._input_x, self._input_t, self._input_y))
            self._data_set = self._data_set.repeat().shuffle(self._config.buffer_size).batch(self._config.batch_size)

            self._iter = self._data_set.make_initializable_iterator()
            self._x_batch, self._t_batch, self._y_batch = self._iter.get_next()

    def _gan_def(self):
        batch_size = tf.shape(self._x_batch)[0]

        def gen(x, t, y):
            with tf.variable_scope("generator"):
                zg = tf.random_uniform([batch_size, self._config.k-1], minval=-1, maxval=1)
                i = tf.concat([x, t, y, zg], -1)
                for _ in range(self._config.hidden_layers-1):
                    i = tf.contrib.layers.fully_connected(i, self._config.hidden_units)
                y_tilde = tf.contrib.layers.fully_connected(i, self._config.k, activation_fn=None)
                return y_tilde

        def dis(x, y, reuse=False):
            with tf.variable_scope("discriminator", reuse=reuse):
                i = tf.concat([x, y], -1)
                for _ in range(self._config.hidden_layers-1):
                    i = tf.contrib.layers.fully_connected(i, self._config.hidden_units)
                logits = tf.contrib.layers.fully_connected(i, self._config.k, activation_fn=None)
                return logits

        with tf.variable_scope('gan'):
            self._y_tilde = gen(self._x_batch, self._t_batch, self._y_batch)
            self._y_bar = self._y_tilde * (1 - self._t_batch) + self._y_batch * self._t_batch
            self._cf_logits = dis(self._x_batch, self._y_bar)
            # self._cf_fake_logits = dis(self._x_batch, self._y_tilde, reuse=True)

    def _loss_def(self):
        with tf.name_scope('loss'):
            with tf.name_scope('g_loss'):
                dis_label = 1 - self._t_batch
                self._g_loss = tf.losses.sigmoid_cross_entropy(dis_label, self._cf_logits)
                # self._g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self._cf_fake_logits),
                #                                                self._cf_fake_logits)
                self._y_tilde_eta = tf.reduce_sum(self._y_tilde * self._t_batch, -1, keepdims=True)
                with tf.name_scope('supervised_loss'):
                    self._g_supervised_loss = tf.losses.mean_squared_error(self._y_batch, self._y_tilde_eta)
                self._cf_g_loss = self._g_loss + self._config.alpha * self._g_supervised_loss
            with tf.name_scope('d_loss'):
                self._cf_d_loss = tf.losses.sigmoid_cross_entropy(self._t_batch, self._cf_logits)

    def _train_def(self):
        g_vars = tf.trainable_variables(self._name + "/gan/generator")
        d_vars = tf.trainable_variables(self._name + "/gan/discriminator")

        self._train_g = tf.train.AdamOptimizer(self._config.learning_rate).minimize(self._cf_g_loss,
                                                                                    var_list=g_vars)
        self._train_d = tf.train.AdamOptimizer(self._config.learning_rate).minimize(self._cf_d_loss,
                                                                                    var_list=d_vars)

    def _summary_def(self):
        self._writer = tf.summary.FileWriter(os.path.join(self._config.cf_save_path, 'log'), self._sess.graph)

    def fit(self, data_set):
        self._sess.run(self._init)
        self._sess.run(self._iter.initializer, feed_dict={self._input_x: data_set.x,
                                                          self._input_t: data_set.t,
                                                          self._input_y: data_set.y})

        for i in range(1, self._config.epochs+1):
            _, cf_d_loss = self._sess.run((self._train_d, self._cf_d_loss))
            _, cf_g_loss = self._sess.run((self._train_g, self._cf_g_loss))

    def gen_y_bar(self, data_set):
        return self._sess.run(self._y_bar, feed_dict={self._x_batch: data_set.x,
                                                      self._t_batch: data_set.t,
                                                      self._y_batch: data_set.y})

    def export(self, path):
        pass


class ITEBlock(BaseModel):
    def __init__(self,
                 config: GANITEConfig,
                 name='ite'):
        super().__init__(config, name)
        self._config.save_path = self._config.ite_save_path

    def _build_graph(self):
        with tf.variable_scope(self._name):
            self._placeholder_def()
            self._set_def()
            self._gan_def()
            self._loss_def()
            self._train_def()

    def _placeholder_def(self):
        with tf.name_scope('input'):
            self._input_x = tf.placeholder(tf.float32, [None, self._config.input_dim], name="input_x")
            self._input_y = tf.placeholder(tf.float32, [None, self._config.k], name='input_y')

    def _set_def(self):
        with tf.name_scope('data_set'):
            self._data_set = tf.data.Dataset.from_tensor_slices((self._input_x, self._input_y))
            self._data_set = self._data_set.repeat().shuffle(self._config.buffer_size).batch(self._config.batch_size)

            self._iter = self._data_set.make_initializable_iterator()
            self._x_batch, self._y_batch = self._iter.get_next()

    def _gan_def(self):
        batch_size = tf.shape(self._x_batch)[0]

        def gen(x):
            with tf.variable_scope('generator'):
                zi = tf.random_uniform([batch_size, self._config.k], minval=-1, maxval=1)
                i = tf.concat([x, zi], -1)
                for _ in range(self._config.hidden_layers-1):
                    i = tf.contrib.layers.fully_connected(i, self._config.hidden_units)
                y_hat = tf.contrib.layers.fully_connected(i, self._config.k, activation_fn=None)
                return y_hat

        def dis(x, y, reuse=False):
            with tf.variable_scope("discriminator", reuse=reuse):
                i = tf.concat([x, y], -1)
                for _ in range(self._config.hidden_layers-1):
                    i = tf.contrib.layers.fully_connected(i, self._config.hidden_units)
                logits = tf.contrib.layers.fully_connected(i, 1, activation_fn=None)
                return logits

        with tf.variable_scope("gan"):
            self._y_hat = gen(self._x_batch)
            self._logits = dis(self._x_batch, self._y_batch)
            self._fake_logits = dis(self._x_batch, self._y_hat, reuse=True)

    def _loss_def(self):
        with tf.name_scope('loss'):
            with tf.name_scope('g_loss'):
                self._g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self._fake_logits),
                                                               self._fake_logits)
                with tf.name_scope('supervised_loss'):
                    if self._config.k == 2:
                        self._supervised_loss = tf.losses.mean_squared_error(self._y_batch[:, 1]-self._y_batch[:, 0],
                                                                             self._y_hat[:, 1]-self._y_hat[:, 0])
                    else:
                        self._supervised_loss = tf.losses.mean_squared_error(self._y_batch, self._y_hat)

                self._ite_g_loss = self._g_loss + self._config.beta * self._supervised_loss

            with tf.name_scope('d_loss'):
                self._ite_d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self._logits), self._logits) + \
                                   tf.losses.sigmoid_cross_entropy(tf.zeros_like(self._fake_logits), self._fake_logits)

    def _train_def(self):
        g_vars = tf.trainable_variables(self._name + "/gan/generator")
        d_vars = tf.trainable_variables(self._name + "/gan/discriminator")

        self._train_g = tf.train.AdamOptimizer().minimize(self._ite_g_loss, var_list=g_vars)
        self._train_d = tf.train.AdamOptimizer().minimize(self._ite_d_loss, var_list=d_vars)

    def _summary_def(self):
        self._writer = tf.summary.FileWriter(os.path.join(self._config.ite_save_path, 'log'), self._sess.graph)

    def fit(self, data_set):
        self._sess.run(self._init)
        self._sess.run(self._iter.initializer, feed_dict={self._input_x: data_set.x,
                                                          self._input_y: data_set.y})

        for i in range(1, self._config.epochs+1):
            _, ite_d_loss = self._sess.run((self._train_d, self._ite_d_loss))
            _, ite_g_loss = self._sess.run((self._train_g, self._ite_g_loss))

    def gen_y_hat(self, data_set):
        return self._sess.run(self._y_hat, feed_dict={self._x_batch: data_set.x})

    def export(self, path):
        # todo: export model
        pass
