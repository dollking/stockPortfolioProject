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


class ModelIndex(object):
    def __init__(self, sess, loop_count):
        self.sess = sess
        self.loop_count = loop_count

        self._init_placeholder()

        print('isOK')

    def _init_placeholder(self):
        self.dropout_rate = tf.placeholder(dtype=tf.float32)
        self.index_input1 = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        self.index_input2 = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        self.index_input3 = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        self.index_input4 = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        self.index_input5 = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        self.index_input6 = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        self.index_input7 = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        self.index_input8 = tf.placeholder(dtype=tf.float32, shape=[None, 5])

    def stem_index(self, data, stem_name):
        with slim.arg_scope([slim.fully_connected]):
            with tf.variable_scope(stem_name, values=[data]):
                fnn = slim.fully_connected(slim.dropout(data), 5, scope='feature_extend_1')
                fnn = slim.fully_connected(slim.dropout(fnn), 5, scope='feature_extend_2')
                fnn = slim.fully_connected(slim.dropout(fnn), 6, scope='feature_extend_3')
                return tf.reshape(slim.fully_connected(slim.dropout(fnn), 8, scope='feature_extend_4'), [-1, 8, 1, 1])

    def inception_a(self, data, scale=1.0, scope=None, reuse=None):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockInceptionA', [data], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(data, 2, [1, 3], scope='Conv2d_0a_1x3')
                    branch_0 = slim.conv2d(branch_0, 2, [3, 1], scope='Conv2d_0a_3x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(data, 2, [3, 3], scope='Conv2d_1a_3x3')
                    branch_1 = slim.conv2d(branch_1, 2, [5, 5], scope='Conv2d_1b_5x5')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(data, 2, [1, 1], scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 3, [3, 3], scope='Conv2d_2b_3x3')

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
                    branch_0 = slim.conv2d(data, 12, [1, 3], scope='Conv2d_0a_1x3')
                    branch_0 = slim.conv2d(branch_0, 12, [3, 1], scope='Conv2d_0a_3x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(data, 8, [1, 1], scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 10, [1, 5], scope='Conv2d_1b_1x5')
                    branch_1 = slim.conv2d(branch_1, 12, [5, 1], scope='Conv2d_1c_5x1')

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
                    branch_0 = slim.conv2d(data, 12, [2, 2], scope='Conv2d_0a_2x2')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(data, 12, [1, 1], scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 14, [1, 2], scope='Conv2d_1b_1x3')
                    branch_1 = slim.conv2d(branch_1, 16, [2, 1], scope='Conv2d_1c_3x1')

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
                    branch_0 = slim.max_pool2d(data, [3, 3], padding="VALID", stride=1, scope='MaxPool2d_0a_3x3')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(data, 5, [3, 3], padding="VALID", stride=1, scope='Conv2d_1a_3x3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(data, 2, [1, 1], scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 2, [3, 3], scope='Conv2d_2b_3x3')
                    branch_2 = slim.conv2d(branch_2, 5, [3, 3], padding="VALID", stride=1, scope='Conv2d_2c_3x3')

                return slim.conv2d(tf.concat(values=[branch_0, branch_1, branch_2], axis=3), 15, [1, 1],
                                   scope='Conv2d_result_1x1')

    def reduction_b(self, data):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(None, 'reduction_B', [data]):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.max_pool2d(data, [3, 3], padding="VALID", stride=2, scope='MaxPool2d_0a_3x3')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(data, 10, [1, 1], scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 20, [3, 3], padding="VALID", stride=2, scope='Conv2d_1b_3x3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(data, 10, [1, 1], scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 20, [3, 3], padding="VALID", stride=2, scope='Conv2d_2b_3x3')

                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.conv2d(data, 10, [1, 1], scope='Conv2d_3a_1x1')
                    branch_3 = slim.conv2d(branch_3, 20, [3, 3], scope='Conv2d_3b_3x3')
                    branch_3 = slim.conv2d(branch_3, 25, [3, 3], padding="VALID", stride=2, scope='Conv2d_3c_3x3')

                return slim.conv2d(tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3), 80, [1, 1],
                                   scope='Conv2d_result_1x1')

    def model(self):
        ################################################################################### index data stemming
        index_data1 = slim.batch_norm(self.stem_index(self.index_input1, 'stem_index1'))
        index_data2 = slim.batch_norm(self.stem_index(self.index_input2, 'stem_index2'))
        index_data3 = slim.batch_norm(self.stem_index(self.index_input3, 'stem_index3'))
        index_data4 = slim.batch_norm(self.stem_index(self.index_input4, 'stem_index4'))
        index_data5 = slim.batch_norm(self.stem_index(self.index_input5, 'stem_index5'))
        index_data6 = slim.batch_norm(self.stem_index(self.index_input6, 'stem_index6'))
        index_data7 = slim.batch_norm(self.stem_index(self.index_input7, 'stem_index7'))
        index_data8 = slim.batch_norm(self.stem_index(self.index_input8, 'stem_index8'))

        index_data = slim.batch_norm(
            tf.concat([index_data1, index_data2, index_data3, index_data4,
                       index_data5, index_data6, index_data7, index_data8], axis=2))  # 3
        print(index_data)

        ################################################################################### inception with index data
        index_inception_a = slim.repeat(index_data, self.loop_count[0], self.inception_a)
        index_reduction_a = self.reduction_a(index_inception_a)
        print(index_reduction_a)

        index_inception_b = slim.repeat(index_reduction_a, self.loop_count[1], self.inception_b)
        index_reduction_b = self.reduction_b(index_inception_b)
        print(index_reduction_b)

        index_inception_c = slim.repeat(index_reduction_b, self.loop_count[2], self.inception_c)
        print(index_inception_c)

        output = slim.max_pool2d(index_inception_c, [2, 2], padding="VALID", stride=1)
        output_index = tf.reshape(output, [-1, output.get_shape()[3]])

        inter_output = slim.max_pool2d(index_reduction_b, [2, 2], padding="VALID", stride=1)
        inter_output_index = tf.reshape(inter_output, [-1, inter_output.get_shape()[3]])
        print(output_index)
        print(inter_output_index)

        return output_index, inter_output_index
