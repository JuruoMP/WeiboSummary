# -*- coding:utf-8 -*-
import os
import sys
import traceback
import pickle
import numpy as np
import tensorflow as tf
from data_util import feature_normalize

DEBUG = False


class LSTMLeaderDetection:
    __singleton = None

    MODEL_PATH = 'LSTM_MODEL/'

    n_samples = 0
    batch_size = 1300
    init_learning_rate = 0.001
    learning_rate_decrease = 0.975
    min_learning_rate = 0.000001
    max_epoch = 100000
    lstm_size = 32
    n_hidden_layer = 2
    n_hidden_size = 32
    TINY = 1e-6

    def __init__(self, feature_dim):
        '''
        self.model_parameters = None
        self.X, self.y = None, None
        self.lstm = None
        self.state = None
        self.output = None
        self.pred = None
        '''
        self.feature_dim = feature_dim
        '''
        self.cost = None
        self.optimizer = None
        self.correct_pred = None
        self.accuracy = None
        '''

    @staticmethod
    def get_instance(feature_dim):
        if LSTMLeaderDetection.__singleton is None:
            LSTMLeaderDetection.__singleton = LSTMLeaderDetection(feature_dim)
        return LSTMLeaderDetection.__singleton

    @staticmethod
    def data_generator(data_set, _batch_size=1):
        '''
        Generate data set with batch size
        :param data_set: [(feature_of_path, label_of_path), ...]
        :param _batch_size: batch_size
        :return: a generator to generate data with batch size
        '''
        '''
        num_bits = 9
        def generate_example(num_bits):
            import random
            def as_bytes(num, final_size):
                res = []
                for _ in range(final_size):
                    res.append(num % 2)
                    num //= 2
                return res
            a = random.randint(0, 2 ** (num_bits - 1) - 1)
            b = random.randint(0, 2 ** (num_bits - 1) - 1)
            res = a + b
            return (as_bytes(a, num_bits),
                    as_bytes(b, num_bits),
                    as_bytes(res, num_bits))
        x = np.empty((num_bits, _batch_size, 2))
        y = np.empty((num_bits, _batch_size, 1))

        while True:
            for i in range(_batch_size):
                a, b, r = generate_example(num_bits)
                x[:, i, 0] = a
                x[:, i, 1] = b
                y[:, i, 0] = r
            yield x, y
        '''
        n_samples = len(data_set)
        demo_feature_of_path = data_set[0][0]
        feature_dim = len(demo_feature_of_path[0])
        pos = 0
        while True:
            Xs, ys = [], []
            max_timestemps = 0
            for i in range(_batch_size):
                assert type(data_set[pos][0]) == list
                assert type(data_set[pos][0][0]) == list
                assert type(data_set[pos][1]) == list
                assert len(data_set[pos][0]) == len(data_set[pos][1])
                max_timestemps = max(max_timestemps, len(data_set[pos][0]))
                # Xs.append(data_set[pos][0])
                Xs.append([x for x in data_set[pos][0]])
                ys.append([[x] for x in data_set[pos][1]])
                pos += 1
                if pos >= n_samples:
                    pos -= n_samples
            retX, rety = [], []
            for path_of_features, path_of_labels in [_ for _ in zip(Xs, ys)]:
                pad_timestemps = max_timestemps - len(path_of_features)
                path_of_features += [[0] * feature_dim] * pad_timestemps
                path_of_labels += [[0]] * pad_timestemps
                retX.append(path_of_features)
                rety.append(path_of_labels)
            retX = np.transpose(np.array(retX), axes=(1, 0, 2))
            rety = np.transpose(np.array(rety), axes=(1, 0, 2))
            yield retX, rety


    @staticmethod
    def get_all_data(data_set):
        '''
        Generate data set of all
        :param data_set: [(feature_of_path1, label_of_path1), {feature_of_path2, label_of_path2} ...]
        :return: ([feature_of_path, ...], [label_of_path, ...])
        '''
        Xs, ys = [], []
        max_timestemps = 0
        demo_feature_of_path = data_set[0][0]
        feature_dim = len(demo_feature_of_path[0])
        for pos in range(len(data_set[0])):
            max_timestemps = max(max_timestemps, len(data_set[0][pos]))
            Xs.append([x for x in data_set[0][pos]])
            ys.append([[x] for x in data_set[1][pos]])
        retX, rety = [], []
        for path_of_features, path_of_labels in [_ for _ in zip(Xs, ys)]:
            pad_timestemps = max_timestemps - len(path_of_features)
            path_of_features += [[0] * feature_dim] * pad_timestemps
            path_of_labels += [[0]] * pad_timestemps
            retX.append(path_of_features)
            rety.append(path_of_labels)
        retX = np.transpose(np.array(retX), axes=(1, 0, 2))
        rety = np.transpose(np.array(rety), axes=(1, 0, 2))
        return retX, rety


    def run(self, mode, feature_set, label_set=None):
        self.build_graph()
        if mode == 'TRAIN':
            try:
                if not feature_set:
                    raise Exception('Data set is not given for training')
                elif not label_set:
                    raise Exception('Label set is not given for training')
                self.train(feature_set, label_set, all_data_set=(feature_set, label_set))
            except:
                traceback.print_exc()
                return -1
            return 1
        elif mode == 'EVAL':
            try:
                if not feature_set:
                    raise Exception('Data set is not given for evaluating')
                # elif not self.model_parameters:
                #     raise Exception('Model: mlp_leader_detection is not trained')
                label_set = self.eval(feature_set)
                return label_set
            except:
                traceback.print_exc()
                return None

    def build_graph(self):
        '''
        2 layers' mlp + bi-classified softmax
        '''
        # self.X = tf.placeholder(tf.float32, [None, LSTMLeaderDetection.batch_size, self.feature_dim])
        # self.y = tf.placeholder(tf.float32, [None, LSTMLeaderDetection.batch_size, 1])
        self.X = tf.placeholder(tf.float32, [None, None, self.feature_dim])
        self.y = tf.placeholder(tf.float32, [None, None, 1])
        # cell = tf.contrib.rnn.BasicRNNCell(LSTMLeaderDetection.lstm_size)
        cell = tf.contrib.rnn.BasicLSTMCell(LSTMLeaderDetection.lstm_size, state_is_tuple=True)
        init_state = cell.zero_state(LSTMLeaderDetection.batch_size, dtype=tf.float32)
        self.output, self.states = tf.nn.dynamic_rnn(cell, self.X, initial_state=init_state, time_major=True)
        self.final_projection = lambda x: tf.contrib.layers.linear(x, num_outputs=1, activation_fn=tf.nn.sigmoid)
        self.pred = tf.map_fn(self.final_projection, self.output)
        self.correct_pred = tf.equal(self.pred, self.y)
        self.accuracy = tf.reduce_mean(tf.cast(tf.abs(self.y - self.pred) < 0.5, tf.float32))

    def train(self, feature_set, label_set, all_data_set=None):
        '''
        Train LSTM leader detection model
        :param feature_set: [[features of path1], [features of path2], ...]
        :param label_set:[[label of path1], [label of path2], ...]
        :param all_data_set:
        :return:
        '''
        feature_set = feature_normalize(feature_set, mode='EVAL')
        data_set = zip(feature_set, label_set)
        if sys.version[0] == '3':
            data_set = [x for x in data_set]
        feature_dim = len(feature_set[0])
        n_samples = len(data_set)
        self.cost = -(self.y * tf.log(self.pred + LSTMLeaderDetection.TINY) +
                      (1.0 - self.y) * tf.log((1.0 - self.pred + LSTMLeaderDetection.TINY)))
        self.cost = tf.reduce_mean(self.cost)
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        # self.cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(LSTMLeaderDetection.init_learning_rate).minimize(self.cost)

        #run(mode='TRAIN', data_set=data_set)
        generator = LSTMLeaderDetection.data_generator(data_set, LSTMLeaderDetection.batch_size)
        init = tf.global_variables_initializer()
        # saver = tf.train.Saver(tf.global_variables())
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            restored = False
            if os.path.exists(LSTMLeaderDetection.MODEL_PATH):
                ckpt = tf.train.get_checkpoint_state(LSTMLeaderDetection.MODEL_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    restored = True
                    print("Model restored.")
            if not restored:
                sess.run(init)
                print("Model initialized.")
            # MODE=Train
            for epoch in range(LSTMLeaderDetection.max_epoch):
                total_cost = 0
                total_batch = int(n_samples / LSTMLeaderDetection.batch_size)
                for i in range(total_batch):
                    Xs, ys = next(generator)
                    if DEBUG:
                        print(len(Xs))
                        print(len(Xs[0]))
                        pred_result = sess.run(self.pred, feed_dict={self.X: Xs})
                        print([_ for _ in zip(pred_result, Xs)])
                        _ = input('pause')
                    acc, _, c = sess.run([self.accuracy, self.optimizer, self.cost], feed_dict={self.X: Xs, self.y: ys})
                    total_cost += c

                if epoch % 100 == 0 and all_data_set:
                #    all_Xs, all_ys = LSTMLeaderDetection.get_all_data(all_data_set)
                #    pred_result = sess.run(self.pred, feed_dict={self.X: all_Xs})
                #    acc = self.accuracy.eval({self.X: all_Xs, self.y: all_ys})
                    print('Epoch: %d, loss %.4f, acc %.4f' % (epoch, total_cost, acc))

                if epoch % 1000 == 0:
                    # save_path = saver.save(sess, MlpLeaderDetection.MODEL_PATH + 'mlp_model.cpkt', global_step=epoch//1000)
                    save_path = saver.save(sess, LSTMLeaderDetection.MODEL_PATH + 'lstm_model.cpkt')
                    print("Model saved in file: %s" % save_path)

    def eval(self, feature_set):
        feature_set = feature_normalize(feature_set, mode='EVAL')
        n_samples = len(feature_set)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            restored = False
            if os.path.exists(LSTMLeaderDetection.MODEL_PATH):
                ckpt = tf.train.get_checkpoint_state(LSTMLeaderDetection.MODEL_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    restored = True
                    print("Model restored.")
            if not restored:
                raise Exception("Error: Model file not exist!")
            # else:   # TODO: finish here
            #     exit(-1)
            pred_results = []
            for eval_features in feature_set:
                pred_result = sess.run(self.pred, feed_dict={self.X: np.array([eval_features])})
                pred_results.append(pred_result)
            return pred_results
