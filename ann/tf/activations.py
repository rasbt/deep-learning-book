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
from tensorflow.python.framework import ops


def leaky_relu(x, alpha=0.0001):
    with ops.name_scope('leaky_relu') as scope:
        return tf.maximum(alpha * x, x)


def selu(x):
    # Based on "Self-normalizing networks"(SNNs)
    # Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter
    # https://arxiv.org/abs/1706.02515
    with ops.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))
