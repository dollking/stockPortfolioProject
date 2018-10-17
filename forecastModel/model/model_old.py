"""
	Deep learning project for predicting stock trend with tensorflow.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Class file for session.

	:copyright: Hwang.S.J.
"""
import os
import pickle
import numpy as np
import tensorflow as tf
from random import shuffle

from .model_index import ModelIndex
from .model_title import ModelTitle

slim = tf.contrib.slim


class Model(object):
    def __init__(self, sess, root_dir, index_model_loop_count, title_model_loop_count):
        self.sess = sess
        self.root_dir = root_dir
        self.epoch = 600
        self.dataPath = os.path.join(self.root_dir, 'data', 'data')
        self.index_model = ModelIndex(self.sess, index_model_loop_count)
        self.title_model = ModelTitle(self.sess, title_model_loop_count)

        self._init_placeholder()

        self.learningRate = 0.001
        self.past = 0.0
        self.cnt = 0
        self.max_cnt = 0

        self.best = 0.0
        self.training_best = 0.0
        self.validation_best = 0.0
        self.test_best = 0.0
        self.temp_size = 0
        self.model()

        print('isOK')

    def _init_placeholder(self):
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    def load_data(self, root_data_path, company_name):
        fp = open(os.path.join(root_data_path, 'title', company_name + '.pkl'), 'rb')
        title_data = [np.array(data) for data in pickle.load(fp)]
        fp.close()

        fp = open(os.path.join(root_data_path, 'index', company_name + '.pkl'), 'rb')
        index_data = [np.array(data) for data in pickle.load(fp)]
        fp.close()

        fp = open(os.path.join(root_data_path, 'target', company_name + '.pkl'), 'rb')
        target = [np.array(data) for data in pickle.load(fp)]
        fp.close()

        return index_data, title_data, target

    def model(self):
        output_index, inter_output_index = self.index_model.model()
        output_title, inter_output_title = self.title_model.model()

        ################################################################################### make hypothesis
        with tf.variable_scope('hypothesis_layer'):
            output = tf.concat([output_index, output_title], axis=1)
            print(output)
            fnn = slim.fully_connected(output, 120)
            fnn = slim.fully_connected(slim.dropout(fnn, keep_prob=self.index_model.dropout_rate), 60)
            fnn = slim.fully_connected(slim.dropout(fnn, keep_prob=self.index_model.dropout_rate), 20)
            fnn = slim.fully_connected(slim.dropout(fnn, keep_prob=self.index_model.dropout_rate), 8)
            self.hypothesis = slim.fully_connected(fnn, 2, activation_fn=tf.nn.softmax)

        with tf.variable_scope('inter_hypothesis_layer'):
            inter_output = tf.concat([inter_output_index, inter_output_title], axis=1)
            print(inter_output)
            fnn = slim.fully_connected(inter_output, 120)
            fnn = slim.fully_connected(slim.dropout(fnn, keep_prob=self.index_model.dropout_rate), 60)
            fnn = slim.fully_connected(slim.dropout(fnn, keep_prob=self.index_model.dropout_rate), 20)
            fnn = slim.fully_connected(slim.dropout(fnn, keep_prob=self.index_model.dropout_rate), 8)

            self.inter_hypothesis = slim.fully_connected(fnn, 2, activation_fn=tf.nn.softmax)

            ###################################################################################  loss/optimize
        self.cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis,
                                                                               labels=self.target)) * 0.8) + \
                    (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.inter_hypothesis,
                                                                               labels=self.target)) * 0.2)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost)

        ###################################################################################  accuracy
        self.h_argmax = tf.argmax(self.hypothesis, 1)
        self.t_argmax = tf.argmax(self.target, 1)

        correct_prediction = tf.equal(self.h_argmax, self.t_argmax)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def forecast(self, root_data_path, model_path, company_name):
        saver = tf.train.Saver()
        print(os.path.join(model_path, company_name))
        checkpoint = tf.train.get_checkpoint_state(os.path.join(model_path, company_name))

        saver.restore(self.sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        index, title, target = self.load_data(root_data_path, company_name)

        return self.get_accuracy(index, title, target)

    def get_accuracy(self, index_data, title_data, target):
        hypothesis, acc, h_arg, t_arg = self.sess.run([self.hypothesis, self.acc, self.h_argmax, self.t_argmax],
                                                      feed_dict={self.title_model.title_input1: title_data[0],
                                                                 self.title_model.title_input2: title_data[1],
                                                                 self.title_model.title_input3: title_data[2],
                                                                 self.title_model.title_input4: title_data[3],
                                                                 self.title_model.title_input5: title_data[4],
                                                                 self.title_model.title_input6: title_data[5],
                                                                 self.index_model.index_input1: index_data[0],
                                                                 self.index_model.index_input2: index_data[1],
                                                                 self.index_model.index_input3: index_data[2],
                                                                 self.index_model.index_input4: index_data[3],
                                                                 self.index_model.index_input5: index_data[4],
                                                                 self.index_model.index_input6: index_data[5],
                                                                 self.index_model.index_input7: index_data[6],
                                                                 self.index_model.index_input8: index_data[7],
                                                                 self.target: target,
                                                                 self.index_model.dropout_rate: 1.0})
        h_arg, t_arg = list(h_arg), list(t_arg)
        cnt = sum([(a == 1 and a == b) for a, b in zip(h_arg, t_arg)])

        return hypothesis, acc, cnt / t_arg.count(1), (cnt / h_arg.count(1) if h_arg.count(1) else 0.)

    def train(self, index_data, title_data, target, dropout_rate=0.7):
        return self.sess.run([self.cost, self.opt],
                             feed_dict={self.title_model.title_input1: title_data[0],
                                        self.title_model.title_input2: title_data[1],
                                        self.title_model.title_input3: title_data[2],
                                        self.title_model.title_input4: title_data[3],
                                        self.title_model.title_input5: title_data[4],
                                        self.title_model.title_input6: title_data[5],
                                        self.index_model.index_input1: index_data[0],
                                        self.index_model.index_input2: index_data[1],
                                        self.index_model.index_input3: index_data[2],
                                        self.index_model.index_input4: index_data[3],
                                        self.index_model.index_input5: index_data[4],
                                        self.index_model.index_input6: index_data[5],
                                        self.index_model.index_input7: index_data[6],
                                        self.index_model.index_input8: index_data[7],
                                        self.target: target, self.index_model.dropout_rate: dropout_rate})

    def test(self, company_name):
        index_data, title_data, target = self.load_data(os.path.join(self.dataPath, 'test'), company_name)
        _, total_acc, _, _ = self.get_accuracy(index_data, title_data, target)

        return total_acc

    def training(self, company_name):
        tf.set_random_seed(410)  # reproducibility

        figure_fp = open('figure.txt', 'a')

        model_save_path = os.path.join(self.root_dir, 'trainedModel', company_name)
        os.mkdir(model_save_path)

        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        index_data, title_data, target = self.load_data(os.path.join(self.dataPath, 'train'), company_name)
        self.temp_size = int(len(target) * 0.7)
        for ep in range(self.epoch):
            training_loss, _ = self.train(index_data, title_data, target)

            ########################################################################################################
            #  check accuracy
            _, training_acc, _, _ = self.get_accuracy([data[:self.temp_size] for data in index_data],
                                                      [data[:self.temp_size] for data in title_data],
                                                      [data[:self.temp_size] for data in target])

            if self.past > training_acc and self.max_cnt < 4:
                if self.cnt == 2:
                    self.cnt = 0
                    self.learningRate /= 2
                    self.max_cnt += 1
                    print('learningRate:', self.learningRate)
                else:
                    self.cnt += 1
            else:
                self.cnt = 0
            self.past = training_acc

            _, validation_acc, recall, precision = self.get_accuracy([data[self.temp_size:] for data in index_data],
                                                                     [data[self.temp_size:] for data in title_data],
                                                                     [data[self.temp_size:] for data in target])

            test_acc = self.test()
            if self.best < validation_acc + training_acc and (
                    abs(test_acc - training_acc) < 0.1 or test_acc > 0.6) and (recall > 0 and precision > 0):
                print('save point result :', training_acc, validation_acc, test_acc)
                saver.save(self.sess, os.path.join(model_save_path, 'model'))
                self.training_best = training_acc
                self.validation_best = validation_acc
                self.test_best = test_acc
                self.best = validation_acc + training_acc

            if not (ep + 1) % 5:
                print('epoch{} :'.format(ep + 1), training_acc, validation_acc, test_acc, recall, precision)

        figure_fp.write('{}, {}, {}, {}\n'.format(company_name, self.training_best, self.validation_best, self.test_best))
