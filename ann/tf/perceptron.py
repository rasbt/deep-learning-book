""" TensorFlow Perceptron Graph
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


def perceptron(num_features):
    """TensorFlow graph for training a perceptron

    Parameters
    ----------
    num_features : int
        Number of features in the training dataset

    Graph Operations
    ----------------
    'train:0': Operation for training the perceptron
    'predict:0': Operation for making predictions
    'weights:0': Tensor containing the model weights
    'bias:0': Tensor containing the model's bias

    Example
    -------

    # Add the perceptron to a new graph

    g = tf.Graph()
    with g.as_default() as g:
        perceptron(num_features=num_features)

    # Training and prediction
    # X is a NumPy array with shape [n_samples, n_features]
    # y is a NumPy array with shape [n_samples,]

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(5):
            for example, target in zip(X, y):
                feed_dict = {'features:0': example.reshape(-1, num_features),
                             'targets:0': target.reshape(-1, 1)}
                _ = sess.run('train:0', feed_dict=feed_dict)

        pred = sess.run('predict:0', feed_dict={'features:0': X})
        train_errors = np.sum(pred.reshape(-1) != y)
        print('Number of training errors', train_errors)

        weights, bias = sess.run(['weights:0', 'bias:0'])
        print('Weights:', weights)
        print('Bias:', bias)
    """

    # initialize model parameters
    features = tf.placeholder(dtype=tf.float32,
                              shape=[None, num_features], name='features')
    targets = tf.placeholder(dtype=tf.float32,
                             shape=[None, 1], name='targets')
    params = {
        'weights': tf.Variable(tf.zeros(shape=[num_features, 1],
                                        dtype=tf.float32), name='weights'),
        'bias': tf.Variable([[0.]], dtype=tf.float32, name='bias')}

    # forward pass
    linear = tf.matmul(features, params['weights']) + params['bias']
    ones = tf.ones(shape=tf.shape(linear))
    zeros = tf.zeros(shape=tf.shape(linear))
    predict = tf.where(tf.less(linear, 0.), zeros, ones, name='predict')

    # weight update
    diff = targets - predict

    weight_update = tf.assign_add(params['weights'],
                                  tf.reshape(diff * features,
                                             (num_features, 1)))

    # workaround to update both weights and bias via one operation
    with tf.control_dependencies([weight_update]):
        train = tf.assign_add(params['bias'], diff, name='train')
