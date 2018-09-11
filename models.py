import tensorflow as tf


class ModelConfig(object):
    def __init__(self):
        self.learning_rate = 0.001
        self.exponential_decay = True
        self.decay_rate = 0.99
        self.decay_steps = 10

        self.l2_of_p = 0  # l2 normalization of predictor
        self.l2_of_d = 0  # l2 normalization of discriminator

        self.epochs = 1000

        self.input_num = 100
        self.output_num = 1

        self.save_path = './model_save'
        self.save_n_times = 10

    @property
    def save_n_steps(self):
        return int(self.epochs / self.save_n_times)


class Predictor(object):
    def __init__(self, name):
        self._name = name

    def __call__(self, inputs, output_num, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            logits = tf.contrib.layers.fully_connected(inputs, output_num, activation_fn=None)

        return logits


class Discriminator(object):
    def __init__(self, name):
        self._name = name

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            logits = tf.contrib.layers.fully_connected(inputs, 1, activation_fn=None)

        return logits


class Model(object):
    def __init__(self,
                 config: ModelConfig,
                 name='model'):
        self._config = config
        self._name = name
        self._predictor = Predictor('predictor')
        self._discriminator = Discriminator('discriminator')

    def build(self):
        with tf.variable_scope(self._name):
            self._placeholder()
            self._pred_forward()

    def _placeholder(self):
        self._input = tf.placeholder(tf.float32, [None, self._config.input_num], name='input_x')
        self._fake_input = tf.placeholder(tf.float32, [None, self._config.input_num], name="fake_input_x")
        self._true_label = tf.placeholder(tf.float32, [None, self._config.output_num], name='true_label')
        self._global_step = tf.get_variable('global_step',
                                            [],
                                            dtype=tf.int32,
                                            trainable=False,
                                            initializer=tf.zeros_initializer())
        if self._config.exponential_decay:
            self._learning_rate = tf.train.exponential_decay(self._config.learning_rate,
                                                             self._global_step,
                                                             self._config.decay_steps,
                                                             self._config.decay_rate)
        else:
            self._learning_rate = tf.constant(self._config.learning_rate)

    def _pred_forward(self):
        self._logits = self._predictor(self._input, self._config.output_num)
        self._fake_logits = self._predictor(self._fake_input, self._config.output_num, reuse=True)

    def _dis_forward(self):
        # todo: change the input to discriminator
        self._dis = self._discriminator(self._input)
        self._fake_dis = self._discriminator(self._fake_input, reuse=True)

    def _pred_def(self):
        # todo: def personal treatment effect
        self._pte = tf.nn.sigmoid(self._logits) - tf.nn.sigmoid(self._fake_logits)

    def _loss_def(self):
        # use train data to train predictor
        self._predict_loss = tf.losses.sigmoid_cross_entropy(self._true_label, self._logits)
        if self._config.l2_of_p:
            prl = self._config.l2_of_p * tf.losses.get_regularization_loss(self._name + '/predictor')
            self._predict_loss = self._predict_loss + prl

        # adversarial loss
        self._g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self._fake_dis), self._fake_dis)
        self._d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self._dis), self._dis) + \
            tf.losses.sigmoid_cross_entropy(tf.zeros_like(self._fake_dis), self._fake_dis)
        if self._config.l2_of_d:
            drl = self._config.l2_of_d * tf.losses.get_regularization_loss(self._name + '/discriminator')
            self._d_loss = self._d_loss + drl

    def _train_def(self):
        # train predictor
        self._train_predictor_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self._predict_loss)

        # adversarial training
        g_vars = tf.trainable_variables(self._name + '/predictor')
        d_vars = tf.trainable_variables(self._name + '/discriminator')
        self._train_g = tf.train.AdamOptimizer(self._learning_rate).minimize(self._g_loss,
                                                                             var_list=g_vars)
        self._train_d = tf.train.AdamOptimizer(self._learning_rate).minimize(self._d_loss,
                                                                             var_list=d_vars,
                                                                             global_step=self._global_step)

    def _init_session(self):
        c = tf.ConfigProto()
        c.gpu_options.allow_growth = True
        self._sess = tf.Session(config=c)
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver(max_to_keep=self._config.save_n_times)

    def fit(self):
        # todo: train predictor

        # todo: train discriminator

        # todo: adversarial and reinforcement training

        pass

    def predict(self):
        # todo: give the PTE
        pass

    def export(self, path):
        """export model with input: true state and treatment, output: PTE

        :param path: path to saved model
        """
        builder = tf.saved_model.builder.SavedModelBuilder(path)
        input_info = tf.saved_model.utils.build_tensor_info(self._input)
        fake_input_info = tf.saved_model.utils.build_tensor_info(self._fake_input)
        output_info = tf.saved_model.utils.build_tensor_info(self._pte)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input': input_info,
                        'fake_input': fake_input_info},
                outputs={'pte': output_info},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        builder.add_meta_graph_and_variables(self._sess,
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={
                                                 'predict': prediction_signature
                                             })
        builder.save()