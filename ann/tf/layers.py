""" Simplified TensorFlow layers for reuse
"""
# Sebastian Raschka 2016-2017
#
# ann is a supporting package for the book
# "Introduction to Artificial Neural Networks and Deep Learning:
#  A Practical Guide with Applications in Python"
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import tensorflow as tf


def conv2d(input_tensor, output_channels,
           kernel_size=(5, 5), strides=(1, 1, 1, 1),
           padding='SAME', activation=None, seed=None,
           name='conv2d'):

    with tf.name_scope(name):
        input_channels = input_tensor.get_shape().as_list()[-1]
        weights_shape = (kernel_size[0], kernel_size[1],
                         input_channels, output_channels)

        weights = tf.Variable(tf.truncated_normal(shape=weights_shape,
                                                  mean=0.0,
                                                  stddev=0.1,
                                                  dtype=tf.float32,
                                                  seed=seed),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=(output_channels,)), name='biases')
        conv = tf.nn.conv2d(input=input_tensor,
                            filter=weights,
                            strides=strides,
                            padding=padding)

        act = conv + biases
        if activation is not None:
            act = activation(conv + biases)
        return act


def fully_connected(input_tensor, output_nodes,
                    activation=None, seed=None,
                    name='fully_connected'):

    with tf.name_scope(name):
        input_nodes = input_tensor.get_shape().as_list()[1]
        weights = tf.Variable(tf.truncated_normal(shape=(input_nodes,
                                                         output_nodes),
                                                  mean=0.0,
                                                  stddev=0.1,
                                                  dtype=tf.float32,
                                                  seed=seed),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[output_nodes]), name='biases')

        act = tf.matmul(input_tensor, weights) + biases
        if activation is not None:
            act = activation(act)
        return act
