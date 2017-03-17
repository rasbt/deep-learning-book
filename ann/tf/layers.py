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


def conv_layer(input, input_channels, output_channels,
               kernel_dim=[5, 5], strides=[1, 1, 1, 1],
               activation=None, seed=None, name='conv'):

    with tf.name_scope(name):
        weights_shape = kernel_dim + [input_channels, output_channels]
        weights = tf.Variable(tf.truncated_normal(shape=weights_shape,
                                                  mean=0.0,
                                                  stddev=0.1,
                                                  dtype=tf.float32,
                                                  seed=seed),
                              name='Weights')
        biases = tf.Variable(tf.zeros(shape=[output_channels]), name='Biases')
        conv = tf.nn.conv2d(input=input,
                            filter=weights,
                            strides=strides,
                            padding='SAME')

        if activation is False:
            act = conv + biases
        else:
            if activation is None:
                activation = tf.nn.relu
            act = activation(conv + biases)
        return act


def fc_layer(input, input_nodes, output_nodes,
             activation=None, seed=None, name='fc'):

    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal(shape=[input_nodes,
                                                         output_nodes],
                                                  mean=0.0,
                                                  stddev=0.1,
                                                  dtype=tf.float32,
                                                  seed=seed),
                              name='Weights')
        biases = tf.Variable(tf.zeros(shape=[output_nodes]), name='Biases')

        if activation is False:
            act = tf.matmul(input, weights) + biases
        else:
            if activation is None:
                activation = tf.nn.relu
            act = activation(tf.matmul(input, weights) + biases)
        return act
