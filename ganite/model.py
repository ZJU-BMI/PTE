import os

import tensorflow as tf

from models import BaseModel


class GANITEConfig(object):
    def __init__(self):
        self.k = 2
        self.input_dim = 25

        self.alpha = 0.1
        self.beta = 0.1

        self.epochs = 1000
        self.learning_rate = 0.001
        self.batch_size = 128

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

            def gen(x, t, y):
                with tf.variable_scope('generator'):
                    zg = tf.random_uniform([batch_size, self._config.k-1], minval=-1, maxval=1)
                    i = tf.concat([x, t, y, zg], 1)
                    y_tilde = tf.contrib.layers.fully_connected(i, self._config.k)
                    return y_tilde

            def dis(x, y, reuse=False):
                with tf.variable_scope('discriminator', reuse=reuse):
                    i = tf.concat([x, y], 1)
                    logits = tf.contrib.layers.fully_connected(i, self._config.k)
                    return logits

            self._y_tilde = gen(self._input_x, self._input_t, self._input_y)
            self._y_bar = self._input_t * self._input_y + (1 - self._input_t) * self._y_tilde
            # self._f_logits = dis(self._input_x, self._input_y)
            self._cf_logits = dis(self._input_x, self._y_bar)

    def _ite_block(self):
        with tf.variable_scope('ite'):
            batch_size = tf.shape(self._input_x)[0]

            def gen(x):
                with tf.variable_scope('generator'):
                    zi = tf.random_uniform([batch_size, self._config.k], minval=-1, maxval=1)
                    zi = tf.placeholder_with_default(zi, [None, self._config.k], name='zi')
                    i = tf.concat([x, zi], 1)
                    y_tilde = tf.contrib.layers.fully_connected(i, self._config.k)
                    return y_tilde

            def dis(x, y, reuse=False):
                with tf.variable_scope('discriminator', reuse=reuse):
                    i = tf.concat([x, y], 1)
                    logits = tf.contrib.layers.fully_connected(i, 1)
                    return logits

            self._y_hat = gen(self._input_x)
            # self._zi = tf.get_default_graph().get_tensor_by_name(f'{self._name}/ite/generator/zi:0')
            self._y_star = dis(self._input_x, self._y_bar)
            self._y_star_fake = dis(self._input_x, self._y_hat, True)

    def _loss_def(self):
        with tf.variable_scope('cf'):
            with tf.name_scope('g_loss'):
                dis_label = tf.ones_like(self._cf_logits) - self._input_t
                self._cf_g_loss = tf.losses.sigmoid_cross_entropy(dis_label,
                                                                  self._cf_logits)
                y_tilde_eta = tf.reduce_sum(self._y_tilde * self._input_x, -1)
                with tf.name_scope('supervised_loss'):
                    self._cf_g_supervised_loss = tf.losses.mean_squared_error(self._input_y, y_tilde_eta)
                self._cf_loss = self._cf_g_loss + self._config.alpha * self._cf_g_supervised_loss

            with tf.name_scope('d_loss'):
                self._cf_d_loss = tf.losses.sigmoid_cross_entropy(1-dis_label, self._cf_logits)

        with tf.variable_scope('ite'):
            with tf.name_scope('g_loss'):
                self._ite_g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self._y_star_fake))
                with tf.name_scope('supervised_loss'):
                    if self._config.k == 2:
                        self._ite_g_supervised_loss = tf.losses.mean_squared_error(self._y_bar[:, 1]-self._y_bar[:, 0],
                                                                                   self._y_hat[:, 1]-self._y_hat[:, 0])
                    else:
                        self._ite_g_supervised_loss = tf.losses.mean_squared_error(self._y_bar, self._y_hat)
                self._ite_loss = self._ite_g_loss + self._config.beta * self._ite_g_supervised_loss

            with tf.name_scope('d_loss'):
                self._ite_d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self._y_star), self._y_star) + \
                                   tf.losses.sigmoid_cross_entropy(tf.zeros_like(self._y_star_fake), self._y_star_fake)

    def _train_def(self):
        cf_g_vars = tf.trainable_variables(self._name + "/cf/generator")
        cf_d_vars = tf.trainable_variables(self._name + "/cf/discriminator")
        ite_g_vars = tf.trainable_variables(self._name + "/ite/generator")
        ite_d_vars = tf.trainable_variables(self._name + "/ite/discriminator")

        # train cf block
        self._train_cf_g = tf.train.AdamOptimizer(self._config.learning_rate).minimize(self._cf_g_loss,
                                                                                       var_list=cf_g_vars)
        self._train_cf_d = tf.train.AdamOptimizer(self._config.learning_rate).minimize(self._cf_d_loss,
                                                                                       var_list=cf_d_vars)

        # train ite block
        self._train_ite_g = tf.train.AdamOptimizer(self._config.learning_rate).minimize(self._ite_g_loss,
                                                                                        var_list=ite_g_vars)
        self._train_ite_d = tf.train.AdamOptimizer(self._config.learning_rate).minimize(self._ite_d_loss,
                                                                                        var_list=ite_d_vars)

    def _init_session(self):
        c = tf.ConfigProto()
        c.gpu_options.allow_growth = True
        self._sess = tf.Session(config=c)
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver(max_to_keep=self._config.save_n_times)

        self._writer = tf.summary.FileWriter(os.path.join(self._config.save_path, 'log'), self._sess.graph)

    def fit(self):
        # train counter factual block

        # train individual treatment effect block
        pass

    def gen_y_bar(self, x, t, y):
        """generate "complete" data set

        :return: y_bar
        """
        return self._sess.run(self._y_bar, feed_dict={self._input_x: x,
                                                      self._input_t: t,
                                                      self._input_y: y})

    def ite_pred(self, x):
        return self._sess.run(self._y_hat, feed_dict={self._input_x: x})

    def save(self, path, step):
        self._saver.save(self._sess, path, step)

    def export(self, path):
        builder = tf.saved_model.builder.SavedModelBuilder(path)
        input_info = tf.saved_model.utils.build_tensor_info(self._input_x)
        ite_info = tf.saved_model.utils.build_tensor_info(self._y_hat)

        ite_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input': input_info},
                outputs={'ite': ite_info},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        builder.add_meta_graph_and_variables(self._sess,
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={
                                                 'predict': ite_signature
                                             })

        builder.save()


class CFBlockConfig(object):
    def __init__(self):
        self.k = 2
        self.input_dim = 25

        self.hidden_unit = 8
        self.hidden_layers = 5

        self.alpha = 2

        self.epochs = 1000
        self.learning_rate = 0.001
        self.batch_size = 64

        self.save_n_times = 1
        self.save_path = './model_save/ganite/cfblock'

    @property
    def buffer_size(self):
        return self.batch_size * 10

    @property
    def save_n_epochs(self):
        return int (self.epochs / self.save_n_times)


class CFBlock(BaseModel):
    def __init__(self,
                 config: CFBlockConfig,
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
                for _ in range(self._config.hidden_layers):
                    i = tf.contrib.layers.fully_connected(i, self._config.hidden_units)
                y_tilde = tf.contrib.layers.fully_connected(i, self._config.k)
                return y_tilde

        def dis(x, y, reuse=False):
            with tf.variable_scope("discriminator", reuse=reuse):
                i = tf.concat([x, y], -1)
                for _ in range(self._config.hidden_layers):
                    i = tf.contrib.layers.fully_connected(i, self._config.hidden_units)
                logits = tf.contrib.layers.fully_connected(i, self._config.k)
                return logits

        with tf.name_scope('gan'):
            self._y_tilde = gen(self._x_batch, self._t_batch, self._y_batch)
            self._y_bar = self._y_tilde * (1 - self._t_batch) + self._y_batch * self._t_batch
            self._cf_logits = dis(self._x_batch, self._y_bar)

    def _loss_def(self):
        with tf.name_scope('loss'):
            with tf.name_scope('g_loss'):
                dis_label = tf.ones_like(self._cf_logits) - self._t_batch
                self._g_loss = tf.losses.sigmoid_cross_entropy(dis_label, self._cf_logits)
                y_tilde_eta = tf.reduce_sum(self._y_tilde * self._x_batch, -1)
                with tf.name_scope('supervised_loss'):
                    self._g_supervised_loss = tf.losses.mean_squared_error(self._y_batch, y_tilde_eta)
                self._cf_g_loss = self._g_loss + self._config.alpha * self._g_supervised_loss
            with tf.name_scope('d_loss'):
                self._cf_d_loss = tf.losses.sigmoid_cross_entropy(1-dis_label, self._cf_logits)

    def _train_def(self):
        g_vars = tf.trainable_variables(self._name + "/gan/generator")
        d_vars = tf.trainable_variables(self._name + "/gan/discriminator")

        self._train_g = tf.train.AdamOptimizer(self._config.learning_rate).minimize(self._cf_g_loss,
                                                                                    var_list=g_vars)
        self._train_d = tf.train.AdamOptimizer(self._config.learning_rate).minimize(self._cf_d_loss,
                                                                                    var_list=d_vars)

    def fit(self, data_set):
        self._sess.run(self._iter.initializer, feed_dict={self._input_x: data_set.x,
                                                          self._input_t: data_set.t,
                                                          self._input_y: data_set.y})

        for i in range(self._config.epochs):
            self._sess.run(self._train_d)
            self._sess.run(self._train_g)

    def gen_y_bar(self, data_set):
        return self._sess.run(self._y_bar, feed_dict={self._x_batch: data_set.x,
                                                      self._t_batch: data_set.t,
                                                      self._y_batch: data_set.y})

    def export(self, path):
        pass


class ITEBlockConfig(object):
    def __init__(self):
        self.k = 2
        self.input_dim = 25

        self.hidden_unit = 8
        self.hidden_layers = 5

        self.beta = 0.1

        self.epochs = 1000
        self.learning_rate = 0.001
        self.batch_size = 128

        self.save_n_times = 1
        self.save_path = './model_save/ganite/iteblock'


class ITEBlock(BaseModel):
    def __init__(self,
                 config: ITEBlockConfig,
                 name='ite'):
        super().__init__(config, name)

    def _build_graph(self):
        self._placeholder_def()

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
                for _ in range(self._config.hidden_layers):
                    i = tf.contrib.layers.fully_connected(i, self._config.hidden_units)
                y_hat = tf.contrib.layers.fully_connected(i, self._config.k)
                return y_hat

        def dis(x, y, reuse=False):
            with tf.variable_scope("discriminator", reuse=reuse):
                i = tf.concat([x, y], -1)
                for _ in range(self._config.hidden_layers):
                    i = tf.contrib.layers.fully_connected(i, self._config.hidden_units)
                logits = tf.contrib.layers.fully_connected(i, 1)
                return logits

        with tf.name_scope("gan"):
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

    def fit(self, data_set):
        self._sess.run(self._iter.initializer, feed_dict={self._input_x: data_set.x,
                                                          self._input_y: data_set.y})

        for i in range(self._config.epochs):
            self._sess.run(self._train_d)
            self._sess.run(self._train_g)

    def gen_y_hat(self, data_set):
        return self._sess.run(self._y_hat, feed_dict={self._x_batch: data_set.x})

    def export(self, path):
        pass
