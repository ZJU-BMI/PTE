import os

import tensorflow as tf


class GANITEConfig(object):
    def __init__(self):
        self.k = 2
        self.input_dim = 25

        self.alpha = 0.1
        self.beta = 0.1

        self.save_n_times = 1
        self.save_path = './model_save/ganite'


class Model(object):
    """reproduce of the ganite

    | Ref paper `GANITE: Estimation of individual treatment effects using GANs
    <https://openreview.net/pdf?id=ByKWUeWA->`

    """
    def __init__(self,
                 config: GANITEConfig,
                 name='ganite'):
        self._config = config
        self._name = name

    def build(self):
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        with tf.variable_scope(self._name):
            self._placeholder_def()
            self._cf_block()
            self._ite_block()
            self._loss_def()
            self._train_def()

    def _placeholder_def(self):
        with tf.name_scope("input"):
            self._input_x = tf.placeholder(tf.float32, [None, self._config.input_dim], name='input_x')
            self._input_t = tf.placeholder(tf.float32, [None, self._config.k], name='input_t')
            self._input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')

    def _cf_block(self):
        with tf.variable_scope('cf'):
            batch_size = tf.shape(self._input_x)[0]

            def gen(x, t):
                with tf.variable_scope('generator'):
                    zg = tf.random_uniform([batch_size, self._config.k-1], minval=-1, maxval=1)
                    i = tf.concat([x, t, zg], 1)
                    y_tilde = tf.contrib.layers.fully_connected(i, self._config.k)
                    return y_tilde

            def dis(x, y, reuse=False):
                with tf.variable_scope('discriminator', reuse=reuse):
                    i = tf.concat([x, y], 1)
                    logits = tf.contrib.layers.fully_connected(i, self._config.k)
                    return logits

            self._y_tilde = gen(self._input_x, self._input_t)
            self._y_bar = self._input_t * self._input_y + (1 - self._input_t) * self._y_tilde
            self._f_logits = dis(self._input_x, self._input_y)
            self._cf_logits = dis(self._input_x, self._y_tilde, True)

    def _ite_block(self):
        with tf.variable_scope('ite'):
            batch_size = tf.shape(self._input_x)[0]

            def gen(x):
                with tf.variable_scope('generator'):
                    zi = tf.random_uniform([batch_size, self._config.k], minval=-1, maxval=1)
                    i = tf.concat([x, zi], 1)
                    y_tilde = tf.contrib.layers.fully_connected(i, self._config.k)
                    return y_tilde

            def dis(x, y, reuse=False):
                with tf.variable_scope('discriminator', reuse=reuse):
                    i = tf.concat([x, y], 1)
                    logits = tf.contrib.layers.fully_connected(i, 1)
                    return logits

            self._y_hat = gen(self._input_x)
            self._y_star = dis(self._input_x, self._y_bar)
            self._y_star_fake = dis(self._input_x, self._y_hat, True)

    def _loss_def(self):
        with tf.variable_scope('cf'):
            with tf.name_scope('g_loss'):
                dis_label = tf.ones_like(self._cf_logits) - self._input_t
                self._cf_g_loss = tf.losses.sigmoid_cross_entropy(dis_label,
                                                                  self._cf_logits)
                y_tilde_eta = tf.reduce_sum(self._y_tilde * self._input_x, -1)
                self._cf_g_supervised_loss = tf.losses.sigmoid_cross_entropy(self._input_y, y_tilde_eta)
                self._cf_loss = self._cf_g_loss + self._config.alpha * self._cf_g_supervised_loss

            with tf.name_scope('d_loss'):
                self._cf_d_loss = tf.losses.sigmoid_cross_entropy(dis_label, self._f_logits) + \
                                  tf.losses.sigmoid_cross_entropy(1-dis_label, self._cf_logits)
        with tf.variable_scope('ite'):
            with tf.name_scope('g_loss'):
                self._ite_g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self._y_star_fake))
                if self._config.k == 2:
                    self._ite_g_supervised_loss = tf.squared_difference(self._y_bar[1] - self._y_bar[0],
                                                                        self._y_hat[1] - self._y_hat[0])
                else:
                    self._ite_g_supervised_loss = tf.squared_difference(self._y_bar, self._y_hat)
                self._ite_loss = self._ite_g_loss + self._config.beta * self._ite_g_supervised_loss

            with tf.name_scope('d_loss'):
                self._ite_d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self._y_star), self._y_star) + \
                                   tf.losses.sigmoid_cross_entropy(tf.zeros_like(self._y_star_fake), self._y_star_fake)

    def _train_def(self):
        cf_g_vars = tf.trainable_variables(self._name + "cf/generator")
        cf_d_vars = tf.trainable_variables(self._name + "cf/discriminator")
        ite_g_vars = tf.trainable_variables(self._name + "ite/generator")
        ite_d_vars = tf.trainable_variables(self._name + "ite/discriminator")

        # train cf block
        self._train_cf_g = tf.train.AdamOptimizer().minimize(self._cf_g_loss,
                                                             var_list=cf_g_vars)
        self._train_cf_d = tf.train.AdamOptimizer().minimize(self._cf_d_loss,
                                                             var_list=cf_d_vars)

        # train ite block
        self._train_ite_g = tf.train.AdamOptimizer().minimize(self._ite_g_loss,
                                                              var_list=ite_g_vars)
        self._train_ite_d = tf.train.AdamOptimizer().minimize(self._ite_d_loss,
                                                              var_list=ite_d_vars)

    def _init_session(self):
        c = tf.ConfigProto()
        c.gpu_options.allow_growth = True
        self._sess = tf.Session(config=c)
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver(max_to_keep=self._config.save_n_times)

        self._writer = tf.summary.FileWriter(os.path.join(self._config.save_path, 'log'), self._sess.graph)

    def fit(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def export(self):
        pass
