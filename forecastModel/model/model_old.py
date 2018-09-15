"""
	Deep learning project for predicting stock trend with tensorflow.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Class file for session.

	:copyright: Hwang.S.J.
"""
import os
import tensorflow as tf
import pickle
import numpy as np
from random import shuffle

slim = tf.contrib.slim


class Model(object):
    def __init__(self, sess):
        self.sess = sess
        self.current_dir = os.getcwd()
        self.epoch = 600
        self.dataPath = os.path.join(self.current_dir, 'data', 'data')

        self.dropout_rate = tf.placeholder(dtype=tf.float32)
        self.title_input1 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])
        self.title_input2 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])
        self.title_input3 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])
        self.title_input4 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])
        self.title_input5 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])
        self.title_input6 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])
        self.index_input1 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 5])
        self.index_input2 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.index_input3 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.index_input4 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.index_input5 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.index_input6 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, 2])

        self.learningRate = 0.001
        self.past = 0.0
        self.cnt = 0
        self.max_cnt = 0

        self.best = 0.0
        self.training_best = 0.0
        self.validation_best = 0.0
        self.direction_best = 0.0

        self.rnn_unit_size = 13

        self.part_results = [0.0 for _ in range(4)]

        self.model()

        tf.set_random_seed(777)  # reproducibility
        print('isOK')

    def cell(self):
        return tf.contrib.rnn.LSTMCell(num_units=self.rnn_unit_size, activation=tf.tanh)

    def blstm(self, data, scope):
        with tf.variable_scope(scope):
            return tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                [self.cell() for _ in range(10)],
                [self.cell() for _ in range(10)],
                data,
                dtype=tf.float32)[0]

    def stem_title(self, data, size, stem_name):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(stem_name, values=[data, size]):
                data = tf.reshape(data, [-1, 3, 18, 1])
                net = slim.conv2d(data, 1, [3, 3], padding='SAME', scope='Conv2d_0a_3x3')
                net = slim.conv2d(net, 2, [1, 6], padding='SAME', scope='Conv2d_1a_1x6')

                return slim.conv2d(net, 2, [6, 1], padding='SAME', scope='Conv2d_2a_6x1')

    def stem_index(self, data, size, stem_name):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(stem_name, values=[data, size]):
                data = tf.reshape(data, [-1, 26, self.rnn_unit_size * 2, 1])
                net = slim.conv2d(data, 2, [3, 3], padding='SAME', scope='Conv2d_0a_3x3')
                net = slim.conv2d(net, 4, [3, 3], padding='SAME', scope='Conv2d_1a_3x3')

                with tf.variable_scope('Mixed_3a'):
                    branch_0 = slim.conv2d(net, 6, [3, 3], padding='SAME', stride=2, scope='Conv2d_2a_3x3')
                    branch_1 = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool2d_2b_3x3')

                    net = tf.concat(values=[branch_0, branch_1], axis=3)

                with tf.variable_scope('Mixed_4a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 4, [1, 1], padding='SAME', scope='Conv2d_4a_1x1')
                        branch_0 = slim.conv2d(branch_0, 6, [3, 3], padding='SAME', scope='Conv2d_4b_3x3')

                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 4, [1, 1], padding='SAME', scope='Conv2d_4a_1x1')
                        branch_1 = slim.conv2d(branch_1, 4, [7, 1], padding='SAME', scope='Conv2d_4b_7x1')
                        branch_1 = slim.conv2d(branch_1, 4, [1, 7], padding='SAME', scope='Conv2d_4b_1x7')
                        branch_1 = slim.conv2d(branch_1, 6, [3, 3], padding='SAME', scope='Conv2d_4b_3x3')

                    return slim.conv2d(tf.concat(values=[branch_0, branch_1], axis=3), 5, [1, 1], padding='SAME',
                                       scope='Conv2d_return_1x1')

    def inception_a(self, data, scale=1.0, scope=None, reuse=None):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockInceptionA', [data], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(data, 2, [1, 1], scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(data, 2, [1, 1], scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 2, [3, 3], scope='Conv2d_1b_3x3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(data, 2, [1, 1], scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 3, [3, 3], scope='Conv2d_2b_3x3')
                    branch_2 = slim.conv2d(branch_2, 4, [3, 3], scope='Conv2d_2c_3x3')

                mixed = tf.concat(values=[branch_0, branch_1, branch_2], axis=3)
                up = slim.conv2d(mixed, data.get_shape()[3], 1, normalizer_fn=None,
                                 activation_fn=None, scope='Conv2d_1x1')

                scaled_up = up * scale
                data += scaled_up
                return tf.nn.relu(data)

    def inception_b(self, data, scale=1.0, scope=None, reuse=None):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockInceptionB', [data], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(data, 12, [1, 1], scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(data, 8, [1, 1], scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 10, [1, 7], scope='Conv2d_1b_1x7')
                    branch_1 = slim.conv2d(branch_1, 12, [7, 1], scope='Conv2d_1c_7x1')

                mixed = tf.concat(values=[branch_0, branch_1], axis=3)
                up = slim.conv2d(mixed, data.get_shape()[3], 1, normalizer_fn=None,
                                 activation_fn=None, scope='Conv2d_1x1')

                scaled_up = up * scale
                data += scaled_up
                return tf.nn.relu(data)

    def inception_c(self, data, scale=1.0, scope=None, reuse=None):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockInceptionC', [data], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(data, 12, [1, 1], scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(data, 12, [1, 1], scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 14, [1, 3], scope='Conv2d_1b_1x3')
                    branch_1 = slim.conv2d(branch_1, 16, [3, 1], scope='Conv2d_1c_3x1')

                mixed = tf.concat(values=[branch_0, branch_1], axis=3)
                up = slim.conv2d(mixed, data.get_shape()[3], 1, normalizer_fn=None,
                                 activation_fn=None, scope='Conv2d_1x1')

                scaled_up = up * scale
                data += scaled_up
                return tf.nn.relu(data)

    def reduction_a(self, data):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(None, 'reduction_A', [data]):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.max_pool2d(data, [3, 3], padding="VALID", stride=2)

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(data, 24, [3, 3], padding="VALID", stride=2)

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(data, 16, [1, 1])
                    branch_2 = slim.conv2d(branch_2, 16, [3, 3])
                    branch_2 = slim.conv2d(branch_2, 24, [3, 3], padding="VALID", stride=(2, 2))

                return slim.conv2d(tf.concat(values=[branch_0, branch_1, branch_2], axis=3), 52, [1, 1],
                                   scope='Conv2d_result_1x1')

    def reduction_b(self, data):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(None, 'reduction_B', [data]):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.max_pool2d(data, [3, 3], padding="VALID", stride=2, scope='MaxPool2d_0a_3x3')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(data, 16, [1, 1], scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 24, [3, 3], padding="VALID", stride=2, scope='Conv2d_1b_3x3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(data, 16, [1, 1], scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 18, [3, 3], padding="VALID", stride=2, scope='Conv2d_2b_3x3')

                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.conv2d(data, 16, [1, 1], scope='Conv2d_3a_1x1')
                    branch_3 = slim.conv2d(branch_3, 18, [3, 3], scope='Conv2d_3b_3x3')
                    branch_3 = slim.conv2d(branch_3, 20, [3, 3], padding="VALID", stride=2, scope='Conv2d_3c_3x3')

                return slim.conv2d(tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3), 100, [1, 1],
                                   scope='Conv2d_result_1x1')

    def model(self):
        ################################################################################### title data stemming
        title_data1 = self.stem_title(self.title_input1, 18, 'stem_title1')
        title_data2 = self.stem_title(self.title_input2, 18, 'stem_title2')
        title_data3 = self.stem_title(self.title_input3, 18, 'stem_title3')
        title_data4 = self.stem_title(self.title_input4, 18, 'stem_title4')
        title_data5 = self.stem_title(self.title_input5, 18, 'stem_title5')
        title_data6 = self.stem_title(self.title_input6, 18, 'stem_title6')

        title_data = slim.batch_norm(
            tf.concat([title_data1, title_data2, title_data3, title_data4, title_data5, title_data6], axis=1))
        print(title_data)

        ################################################################################### index data stemming
        index_data1 = self.stem_index(slim.batch_norm(self.blstm(self.index_input1, 'blstm_input1')), 5, 'stem_index1')
        index_data2 = self.stem_index(slim.batch_norm(self.blstm(self.index_input2, 'blstm_input2')), 4, 'stem_index2')
        index_data3 = self.stem_index(slim.batch_norm(self.blstm(self.index_input3, 'blstm_input3')), 4, 'stem_index3')
        index_data4 = self.stem_index(slim.batch_norm(self.blstm(self.index_input4, 'blstm_input4')), 4, 'stem_index4')
        index_data5 = self.stem_index(slim.batch_norm(self.blstm(self.index_input5, 'blstm_input5')), 4, 'stem_index5')
        index_data6 = self.stem_index(slim.batch_norm(self.blstm(self.index_input6, 'blstm_input6')), 4, 'stem_index6')

        index_data = slim.batch_norm(
            tf.concat([index_data1, index_data2, index_data3, index_data4, index_data5, index_data6], axis=3))
        print(index_data)

        ################################################################################### inception with title data
        title_inception_a = slim.repeat(title_data, 1, self.inception_a)    ### 2/3/1
        title_reduction_a = self.reduction_a(title_inception_a)
        print(title_reduction_a)

        title_inception_b = slim.repeat(title_reduction_a, 2, self.inception_b)
        title_reduction_b = self.reduction_b(title_inception_b)
        print(title_reduction_b)

        title_inception_c = slim.repeat(title_reduction_b, 1, self.inception_c)
        print(title_inception_c)

        output_title = tf.layers.average_pooling2d(inputs=title_inception_c, pool_size=[3, 3], padding="VALID",
                                                   strides=1)
        output_title = tf.reshape(output_title, [-1, output_title.get_shape()[3]])

        inter_output_title = tf.layers.average_pooling2d(inputs=title_inception_b, pool_size=[3, 3], padding="VALID",
                                                         strides=1)
        inter_output_title = tf.reshape(inter_output_title, [-1, inter_output_title.get_shape()[3]])
        print(output_title)
        print(inter_output_title)

        ################################################################################### inception with index data
        index_inception_a = slim.repeat(index_data, 2, self.inception_a)
        index_reduction_a = self.reduction_a(index_inception_a)
        print(index_reduction_a)

        index_inception_b = slim.repeat(index_reduction_a, 2, self.inception_b)
        index_reduction_b = self.reduction_b(index_inception_b)
        print(index_reduction_b)

        index_inception_c = slim.repeat(index_reduction_b, 1, self.inception_c)
        print(index_inception_c)

        output_index = tf.layers.average_pooling2d(inputs=index_inception_c, pool_size=[2, 2], padding="VALID",
                                                   strides=1)
        output_index = tf.reshape(output_index, [-1, output_index.get_shape()[3]])

        inter_output_index = tf.layers.average_pooling2d(inputs=index_inception_b, pool_size=[2, 2], padding="VALID",
                                                         strides=1)
        inter_output_index = tf.reshape(inter_output_index, [-1, inter_output_index.get_shape()[3]])
        print(output_index)
        print(inter_output_index)

        ################################################################################### make hypothesis
        with tf.variable_scope('hypothesis_layer'):
            output = tf.concat([output_index, output_title], axis=1)

            fnn = slim.fully_connected(output, 120)
            fnn = slim.fully_connected(tf.nn.dropout(fnn, keep_prob=self.dropout_rate), 60)
            fnn = slim.fully_connected(tf.nn.dropout(fnn, keep_prob=self.dropout_rate), 20)
            fnn = slim.fully_connected(tf.nn.dropout(fnn, keep_prob=self.dropout_rate), 8)
            self.hypothesis = slim.fully_connected(fnn, 2, activation_fn=tf.nn.softmax)

        with tf.variable_scope('inter_hypothesis_layer'):
            inter_output = tf.concat([inter_output_index, inter_output_title], axis=1)

            fnn = slim.fully_connected(inter_output, 60)
            fnn = slim.fully_connected(tf.nn.dropout(fnn, keep_prob=self.dropout_rate), 20)
            fnn = slim.fully_connected(tf.nn.dropout(fnn, keep_prob=self.dropout_rate), 8)

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
        return self.sess.run([self.acc, self.h_argmax, self.t_argmax],
                             feed_dict={self.title_input1: title_data[0], self.title_input2: title_data[1],
                                        self.title_input3: title_data[2], self.title_input4: title_data[3],
                                        self.title_input5: title_data[4], self.title_input6: title_data[5],
                                        self.index_input1: index_data[0], self.index_input2: index_data[1],
                                        self.index_input3: index_data[2], self.index_input4: index_data[3],
                                        self.index_input5: index_data[4], self.index_input6: index_data[5],
                                        self.target: target, self.dropout_rate: 1.0})

    def train(self, index_data, title_data, target, dropout_rate=0.7):
        return self.sess.run([self.cost, self.opt],
                             feed_dict={self.title_input1: title_data[0], self.title_input2: title_data[1],
                                        self.title_input3: title_data[2], self.title_input4: title_data[3],
                                        self.title_input5: title_data[4], self.title_input6: title_data[5],
                                        self.index_input1: index_data[0], self.index_input2: index_data[1],
                                        self.index_input3: index_data[2], self.index_input4: index_data[3],
                                        self.index_input5: index_data[4], self.index_input6: index_data[5],
                                        self.target: target, self.dropout_rate: dropout_rate})

    def run(self):
        # saver = tf.train.Saver()
        # checkpoint = tf.train.get_checkpoint_state('trainedModel')

        try:
            # saver.restore(self.sess, checkpoint.model_checkpoint_path)
            # print("Successfully loaded:", checkpoint.model_checkpoint_path)

            result = 0.0
            data_list = os.listdir(os.path.join(self.dataPath, 'test', 'title'))
            for company_index in data_list:
                fp = open(os.path.join(self.dataPath, 'train', 'title', company_index), 'rb')
                title_data = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                fp = open(os.path.join(self.dataPath, 'train', 'index', company_index), 'rb')
                index_data = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                fp = open(os.path.join(self.dataPath, 'train', 'target', company_index), 'rb')
                target = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                acc, harg, targ = self.get_accuracy(index_data, title_data, target)
                result += acc

            return result / len(data_list)
            #
            #
            #
            # result = []
            # data_list = os.listdir(os.path.join(self.dataPath, 'training'))
            # for company_index in data_list:
            #     fp = open(os.path.join(self.dataPath, 'training', company_index), 'rb')
            #     data = [np.array(data) for data in pickle.load(fp)]
            #     fp.close()
            #
            #     acc, harg, targ, _ = self.get_accuracy(data)
            #     result.append([company_index, acc])
            #
            #     result.sort(key=lambda x: x[1], reverse=True)
            # print('test set:', result[:10])
            # print('test set:', result[-10:])

        except Exception as e:
            print(e)

    def training(self):
        figure_fp = open('figure.txt', 'a')

        model_save_path = os.path.join(self.current_dir, 'trainedModel')
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

                acc, _, _ = self.get_accuracy(index_data, title_data, target)
                training_acc += acc

            training_acc /= len(data_list[:70])
            training_loss /= len(data_list[:70])
            if self.past > training_acc and self.max_cnt < 3:
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

                acc, harg, targ = self.get_accuracy(index_data, title_data, target)
                target_list.extend(targ)
                hypothesis_list.extend(harg)
                validation_acc += acc

            validation_acc /= len(data_list[70:])

            figure_fp.write('{}, {}, {}\n'.format(training_acc, training_loss, validation_acc))

            test_acc = self.run()
            if self.best < (validation_acc*0.5 + training_acc*0.5) and test_acc > 0.6:
                print(validation_acc, training_acc, test_acc)
                saver.save(self.sess, os.path.join(model_save_path, 'model'))
                self.training_best = training_acc
                self.validation_best = validation_acc
                # self.part_results = part_results

                self.best = validation_acc * 0.5 + training_acc * 0.5

            shuffle(data_list)

            if not (ep + 1) % 5:
                print(training_acc, validation_acc, test_acc)
