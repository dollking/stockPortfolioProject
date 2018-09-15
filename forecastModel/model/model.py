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
    def __init__(self, sess, root_dir):
        self.sess = sess
        self.root_dir = root_dir
        self.epoch = 600
        self.dataPath = os.path.join(self.root_dir, 'data', 'data')
        self.index_model = ModelIndex(self.sess)
        self.title_model = ModelTitle(self.sess)

        self._init_placeholder()

        self.learningRate = 0.002
        self.past = 0.0
        self.cnt = 0
        self.max_cnt = 0

        self.best = 0.0
        self.training_best = 0.0
        self.validation_best = 0.0

        self.model()

        print('isOK')

    def _init_placeholder(self):
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, 2])

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

    def get_predict(self):
        pass

    def get_accuracy(self, index_data, title_data, target):
        return self.sess.run([self.hypothesis, self.acc, self.h_argmax, self.t_argmax],
                             feed_dict={self.title_model.title_input1: title_data[0], self.title_model.title_input2: title_data[1],
                                        self.title_model.title_input3: title_data[2], self.title_model.title_input4: title_data[3],
                                        self.title_model.title_input5: title_data[4], self.title_model.title_input6: title_data[5],
                                        self.index_model.index_input1: index_data[0], self.index_model.index_input2: index_data[1],
                                        self.index_model.index_input3: index_data[2], self.index_model.index_input4: index_data[3],
                                        self.index_model.index_input5: index_data[4], self.index_model.index_input6: index_data[5],
                                        self.index_model.index_input7: index_data[6], self.index_model.index_input8: index_data[7],
                                        self.target: target, self.index_model.dropout_rate: 1.0})

    def train(self, index_data, title_data, target, dropout_rate=0.7):
        return self.sess.run([self.cost, self.opt],
                             feed_dict={self.title_model.title_input1: title_data[0], self.title_model.title_input2: title_data[1],
                                        self.title_model.title_input3: title_data[2], self.title_model.title_input4: title_data[3],
                                        self.title_model.title_input5: title_data[4], self.title_model.title_input6: title_data[5],
                                        self.index_model.index_input1: index_data[0], self.index_model.index_input2: index_data[1],
                                        self.index_model.index_input3: index_data[2], self.index_model.index_input4: index_data[3],
                                        self.index_model.index_input5: index_data[4], self.index_model.index_input6: index_data[5],
                                        self.index_model.index_input7: index_data[6], self.index_model.index_input8: index_data[7],
                                        self.target: target, self.index_model.dropout_rate: dropout_rate})

    def run(self, isTraining=True):
        dir_name = 'train'
        if not isTraining:
            saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state('trainedModel')
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            dir_name = 'test'
            print("Successfully loaded:", checkpoint.model_checkpoint_path)

        total_acc = 0.0
        data_list = os.listdir(os.path.join(self.dataPath, dir_name, 'index'))
        for company_index in data_list:
            fp = open(os.path.join(self.dataPath, dir_name, 'title', company_index), 'rb')
            title_data = [np.array(data) for data in pickle.load(fp)]
            fp.close()

            fp = open(os.path.join(self.dataPath, dir_name, 'index', company_index), 'rb')
            index_data = [np.array(data) for data in pickle.load(fp)]
            fp.close()

            fp = open(os.path.join(self.dataPath, dir_name, 'target', company_index), 'rb')
            target = [np.array(data) for data in pickle.load(fp)]
            fp.close()

            prediction, acc, harg, targ = self.get_accuracy(index_data, title_data, target)

            total_acc += acc
            if not isTraining:
                tp = 0
                for i in range(len(harg)):
                    if harg[i] == targ[i]:
                        if harg[i] == 1:
                            tp += 1
                print('{} : {}(acc), {}(recall), {}(precision)'.format(company_index.split('.')[0],
                                                                       acc, tp/targ.count(1), tp/harg.count(1)))

        if isTraining:
            return total_acc

    def training(self):
        tf.set_random_seed(410)  # reproducibility

        figure_fp = open('figure.txt', 'a')

        model_save_path = os.path.join(self.root_dir, 'trainedModel')
        os.mkdir(model_save_path)

        data_list = os.listdir(os.path.join(self.dataPath, 'train', 'index'))

        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        for ep in range(self.epoch):
            training_acc = 0.0
            training_loss = 0.0

            for company_index in data_list[:70]:
                fp = open(os.path.join(self.dataPath, 'train', 'title', company_index), 'rb')
                title_data = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                fp = open(os.path.join(self.dataPath, 'train', 'index', company_index), 'rb')
                index_data = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                fp = open(os.path.join(self.dataPath, 'train', 'target', company_index), 'rb')
                target = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                loss, opt = self.train(index_data, title_data, target)
                training_loss += loss

            ########################################################################################################
            #  check accuracy
            for company_index in data_list[:70]:
                fp = open(os.path.join(self.dataPath, 'train', 'title', company_index), 'rb')
                title_data = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                fp = open(os.path.join(self.dataPath, 'train', 'index', company_index), 'rb')
                index_data = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                fp = open(os.path.join(self.dataPath, 'train', 'target', company_index), 'rb')
                target = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                _, acc, _, _ = self.get_accuracy(index_data, title_data, target)
                training_acc += acc

            training_acc /= len(data_list[:70])
            training_loss /= len(data_list[:70])
            if self.past > training_acc and self.max_cnt < 4:
                if self.cnt == 3:
                    self.cnt = 0
                    self.learningRate /= 2
                    self.max_cnt += 1
                    print('learningRate:', self.learningRate)
                else:
                    self.cnt += 1
            else:
                self.cnt = 0
            self.past = training_acc

            validation_acc = 0.0
            target_list, hypothesis_list = list(), list()
            for company_index in data_list[70:]:
                fp = open(os.path.join(self.dataPath, 'train', 'title', company_index), 'rb')
                title_data = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                fp = open(os.path.join(self.dataPath, 'train', 'index', company_index), 'rb')
                index_data = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                fp = open(os.path.join(self.dataPath, 'train', 'target', company_index), 'rb')
                target = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                _, acc, harg, targ = self.get_accuracy(index_data, title_data, target)
                target_list.extend(targ)
                hypothesis_list.extend(harg)
                validation_acc += acc

            validation_acc /= len(data_list[70:])

            test_acc = self.run()
            if self.best < (validation_acc*0.5 + training_acc*0.5) and test_acc > 0.6:
                print('save point result :', validation_acc, training_acc, test_acc)
                saver.save(self.sess, os.path.join(model_save_path, 'model'))
                self.training_best = training_acc
                self.validation_best = validation_acc

                self.best = validation_acc * 0.5 + training_acc * 0.5

            figure_fp.write('{}, {}, {}, {}\n'.format(training_acc, training_loss, validation_acc, test_acc))
            shuffle(data_list)

            if not (ep + 1) % 5:
                print('epoch{} :'.format(ep + 1), training_acc, validation_acc, test_acc)
