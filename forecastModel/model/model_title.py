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

slim = tf.contrib.slim


class ModelTitle(object):
    def __init__(self, sess, loot_count):
        self.sess = sess
        self.loop_count = loot_count

        self._init_placeholder()

        print('isOK')

    def _init_placeholder(self):
        self.dropout_rate = tf.placeholder(dtype=tf.float32)
        self.title_input1 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])
        self.title_input2 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])
        self.title_input3 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])
        self.title_input4 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])
        self.title_input5 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])
        self.title_input6 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 18])

    def stem_title(self, data, size, stem_name):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(stem_name, values=[data, size]):
                data = tf.reshape(data, [-1, 3, 18, 1])
                net = slim.conv2d(data, 1, [3, 3], padding='SAME', scope='Conv2d_0a_3x3')
                net = slim.conv2d(net, 2, [1, 6], padding='SAME', scope='Conv2d_1a_1x6')

                return slim.conv2d(net, 2, [6, 1], padding='SAME', scope='Conv2d_2a_6x1')

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
                    branch_0 = slim.max_pool2d(data, [3, 3], padding="VALID", stride=2, scope='MaxPool2d_0a_3x3')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(data, 24, [3, 3], padding="VALID", stride=2, scope='Conv2d_1a_3x3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(data, 16, [1, 1], scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 16, [3, 3], scope='Conv2d_2b_3x3')
                    branch_2 = slim.conv2d(branch_2, 24, [3, 3], padding="VALID", stride=(2, 2), scope='Conv2d_2c_3x3')

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

                return slim.conv2d(tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3), 80, [1, 1],
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

        ################################################################################### inception with title data
        title_inception_a = slim.repeat(title_data, self.loop_count[0], self.inception_a)    ### 2/3/1
        title_reduction_a = self.reduction_a(title_inception_a)
        print(title_reduction_a)

        title_inception_b = slim.repeat(title_reduction_a, self.loop_count[1], self.inception_b)
        title_reduction_b = self.reduction_b(title_inception_b)
        print(title_reduction_b)

        title_inception_c = slim.repeat(title_reduction_b, self.loop_count[2], self.inception_c)
        print(title_inception_c)

        output_title = tf.layers.average_pooling2d(inputs=title_inception_c, pool_size=[3, 3], padding="VALID",
                                                   strides=1)
        output_title = tf.reshape(output_title, [-1, output_title.get_shape()[3]])

        inter_output_title = tf.layers.average_pooling2d(inputs=title_reduction_b, pool_size=[3, 3], padding="VALID",
                                                         strides=1)
        inter_output_title = tf.reshape(inter_output_title, [-1, inter_output_title.get_shape()[3]])
        print(output_title)
        print(inter_output_title)

        return output_title, inter_output_title