# -*- coding:utf-8 -*-
import os
import sys
import traceback
import pickle
import numpy as np
import tensorflow as tf
from data_util import feature_normalize

DEBUG = False


class MlpLeaderDetection:
    __singleton = None

    MODEL_PATH = 'MLP_MODEL/'

    n_samples = 0
    batch_size = 128
    init_learning_rate = 0.001
    learning_rate_decrease = 0.975
    min_learning_rate = 0.000001
    max_epoch = 100000
    n_hidden_size1 = 32
    n_hidden_size2 = 32

    def __init__(self, feature_dim):
        self.model_parameters = None
        self.X, self.y = None, None
        self.weights, self.biases = {}, {}
        self.output = None
        self.pred = None
        self.feature_dim = feature_dim
        self.cost = None
        self.optimizer = None
        self.correct_pred = None
        self.accuracy = None
        self.learning_rate = MlpLeaderDetection.init_learning_rate

    @staticmethod
    def get_instance(feature_dim):
        if MlpLeaderDetection.__singleton is None:
            MlpLeaderDetection.__singleton = MlpLeaderDetection(feature_dim)
        return MlpLeaderDetection.__singleton

    @staticmethod
    def data_generator(data_set, _batch_size=1):
        n_samples = len(data_set)
        pos = 0
        while True:
            Xs, ys = [], []
            for i in range(_batch_size):
                Xs.append(data_set[pos][0])
                ys.append(data_set[pos][1])
                pos += 1
                if pos >= n_samples:
                    pos -= n_samples
            yield np.array(Xs), np.array(ys)
    '''
    def test_data_generator():
        generator = data_generator(20)
        for _ in range(10):
            print(next(generator))
    #test_data_generator()
    #exit(-1)
    '''

    @staticmethod
    def get_all_data(data_set):
        Xs, ys = [], []
        for data in data_set:
            Xs.append(data[0])
            ys.append(data[1])
        return np.array(Xs), np.array(ys)

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
        self.X = tf.placeholder(tf.float32, [None, self.feature_dim])
        self.y = tf.placeholder(tf.float32, [None, 2])
        '''
        self.weights = {
            'h1': tf.Variable(tf.random_normal(
                [self.feature_dim, MlpLeaderDetection.n_hidden_size1],
                mean=0.0, stddev=0.35)),
            'h2': tf.Variable(tf.random_normal(
                [MlpLeaderDetection.n_hidden_size1, MlpLeaderDetection.n_hidden_size2],
                mean=0.0, stddev=0.35)),
            'h3': tf.Variable(tf.random_normal(
                [MlpLeaderDetection.n_hidden_size2, 2],
                mean=0.0, stddev=0.35)),
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([MlpLeaderDetection.n_hidden_size1])),
            'b2': tf.Variable(tf.random_normal([MlpLeaderDetection.n_hidden_size2])),
            'b3': tf.Variable(tf.random_normal([2])),
        }
        z1 = tf.matmul(self.X, self.weights['h1']) + self.biases['b1']
        a1 = tf.nn.sigmoid(z1)
        z2 = tf.matmul(a1, self.weights['h2']) + self.biases['b2']
        a2 = tf.nn.sigmoid(z2)
        z3 = tf.matmul(a2, self.weights['h3']) + self.biases['b3']   # logics
        '''
        W1 = tf.Variable(tf.random_normal([self.feature_dim, 32], mean=0.0, stddev=0.35))
        b1 = tf.Variable(tf.random_normal([32]))
        W2 = tf.Variable(tf.random_normal([32, 2], mean=0.0, stddev=0.35))
        b2 = tf.Variable(tf.random_normal([2]))
        z3 = tf.matmul(tf.nn.sigmoid(tf.matmul(self.X, W1) + b1), W2) + b2
        self.output = z3
        self.softmax = tf.nn.softmax(self.output)
        self.pred = tf.argmax(z3, 1)   # classify output
        self.correct_pred = tf.equal(self.pred, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    '''
    def update_learning_rate():
        global learning_rate, learning_rate_decrease
        new_learning_rate = tf.mul(learning_rate, learning_rate_decrease)
        update_op = tf.assign(learning_rate, new_learning_rate)
        return update_op
    update_learning_op = update_learning_rate()
    '''

    def train(self, feature_set, label_set, all_data_set=None):
        feature_set = feature_normalize(feature_set, mode='EVAL')
        data_set = zip(feature_set, label_set)
        if sys.version[0] == '3':
            data_set = [x for x in data_set]
        feature_dim = len(feature_set[0])
        n_samples = len(data_set)
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.softmax), reduction_indices=[1]))
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        # self.cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(MlpLeaderDetection.init_learning_rate).minimize(self.cost)

        #run(mode='TRAIN', data_set=data_set)
        generator = MlpLeaderDetection.data_generator(data_set, MlpLeaderDetection.batch_size)
        init = tf.global_variables_initializer()
        # saver = tf.train.Saver(tf.global_variables())
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            restored = False
            if os.path.exists(MlpLeaderDetection.MODEL_PATH):
                ckpt = tf.train.get_checkpoint_state(MlpLeaderDetection.MODEL_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    restored = True
                    print("Model restored.")
            if not restored:
                sess.run(init)
                print("Model initialized.")
            # MODE=Train
            for epoch in range(MlpLeaderDetection.max_epoch):
                total_cost = 0
                total_batch = int(n_samples / MlpLeaderDetection.batch_size)
                for i in range(total_batch):
                    Xs, ys = next(generator)
                    if DEBUG:
                        print(len(Xs))
                        print(len(Xs[0]))
                        pred_result = sess.run(self.pred, feed_dict={self.X: Xs})
                        print([_ for _ in zip(pred_result, Xs)])
                        _ = input('pause')
                    _, c = sess.run([self.optimizer, self.cost], feed_dict={self.X: Xs, self.y: ys})
                    total_cost += c
                if epoch % 100 == 0 and all_data_set:
                    Xs, ys = all_data_set
                    if DEBUG:
                        pred_result = sess.run(self.pred, feed_dict={self.X: Xs})
                        cmp_result = [_ for _ in zip(pred_result, ys)]
                        #print(cmp_result)
                        #print(pred_result)
                        #_ = input("Pause...")
                    acc = self.accuracy.eval({self.X: Xs, self.y: ys})
                    print('Epoch: %d, loss %.4f, acc %.4f' % (epoch, total_cost, acc))
                '''
                if epoch % 500 == 0 and self.learning_rate > MlpLeaderDetection.min_learning_rate:
                    self.learning_rate *= MlpLeaderDetection.learning_rate_decrease
                    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
                '''
                if epoch % 1000 == 0:
                    # save_path = saver.save(sess, MlpLeaderDetection.MODEL_PATH + 'mlp_model.cpkt', global_step=epoch//1000)
                    save_path = saver.save(sess, MlpLeaderDetection.MODEL_PATH + 'mlp_model.cpkt')
                    print("Model saved in file: %s" % save_path)

    def eval(self, feature_set):
        feature_set = feature_normalize(feature_set, mode='EVAL')
        n_samples = len(feature_set)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            restored = False
            if os.path.exists(MlpLeaderDetection.MODEL_PATH):
                ckpt = tf.train.get_checkpoint_state(MlpLeaderDetection.MODEL_PATH)
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
