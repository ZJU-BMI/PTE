import abc
import os

import tensorflow as tf


class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self,
                 config,
                 name):
        self._config = config
        self._name = name
        self._g = tf.Graph()
        self.build()

    def build(self):
        with self._g.as_default():
            self._build_graph()
        self._init_session()
        self._summary_def()

    @abc.abstractmethod
    def _build_graph(self):
        pass

    def _init_session(self):
        c = tf.ConfigProto()
        c.gpu_options.allow_growth = True
        self._sess = tf.Session(graph=self._g,
                                config=c)
        with self._g.as_default():
            self._init = tf.global_variables_initializer()
            self._saver = tf.train.Saver(max_to_keep=self._config.save_n_times)
        self._sess.run(self._init)

    def _summary_def(self):
        self._writer = tf.summary.FileWriter(os.path.join(self._config.save_path, 'log'), self._sess.graph)

    @abc.abstractmethod
    def fit(self, data_set):
        pass

    def save(self, path, step=None):
        self._saver.save(self._sess, path, step)

    def export(self, path):
        pass
