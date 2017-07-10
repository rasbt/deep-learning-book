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


def deconv2d_layer(inputs,
                   output_channels,
                   scope,
                   weights_initializer=None,
                   bias_initializer=None,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   activation=None,
                   padding='SAME'):
    with tf.variable_scope(scope):
        static_input_shape = inputs.get_shape().as_list()
        dynamic_input_shape = tf.shape(inputs)

        # workaround for dynamic batch sizes using a symbolic tensor
        batch_size = dynamic_input_shape[0]

        if weights_initializer is None:
            weights_initializer = tf.truncated_normal_initializer(stddev=0.1)

        if bias_initializer is None:
            bias_initializer = tf.constant_initializer(0.0)

        weights = tf.get_variable(name='weights',
                                  shape=[kernel_size[0],
                                         kernel_size[1],
                                         output_channels,
                                         static_input_shape[3]],
                                  initializer=weights_initializer)

        biases = tf.get_variable(name='biases',
                                 shape=(output_channels,),
                                 initializer=bias_initializer)

        if padding not in ('SAME', 'VALID'):
            raise ValueError('Padding must be "SAME" or "VALID"')

        if padding == 'SAME':
            out_height = dynamic_input_shape[1] * strides[0]
            out_width = dynamic_input_shape[2] * strides[1]
        elif padding == 'VALID':
            out_height = ((dynamic_input_shape[1] - 1) * strides[0] +
                          kernel_size[0])
            out_width = ((dynamic_input_shape[2] - 1) * strides[1] +
                         kernel_size[1])

        out_shape = tf.stack([batch_size, out_height,
                              out_width, output_channels])

        deconv = tf.nn.conv2d_transpose(value=inputs,
                                        filter=weights,
                                        output_shape=out_shape,
                                        strides=(1, strides[0], strides[1], 1),
                                        padding=padding)
        deconv = tf.nn.bias_add(deconv, biases)

        if activation is not None:
            deconv = activation(deconv, name='activation')
    return deconv


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
            act = activation(act, name='activation')
        return act
