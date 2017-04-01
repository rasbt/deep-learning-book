# Sebastian Raschka 2016-2017
#
# ann is a supporting package for the book
# "Introduction to Artificial Neural Networks and Deep Learning:
#  A Practical Guide with Applications in Python"
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import unittest
import os
import inspect
import numpy as np
import tensorflow as tf
from ann.tf.perceptron import perceptron


class TestPerceptron(unittest.TestCase):

    this_dir = os.path.dirname(os.path.abspath(inspect.getfile(
                               inspect.currentframe())))
    iris_fname = os.path.join(this_dir, 'iris_binary.csv')
    data = np.genfromtxt(fname=iris_fname, delimiter=',')
    X, y = data[:, :4], data[:, 4]

    def test_perceptron(self):

        X, y = self.X, self.y
        n_features = X.shape[1]

        g = tf.Graph()
        with g.as_default() as g:
            perceptron(num_features=n_features)

        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(5):
                for example, target in zip(X, y):
                    feed_dict = {'features:0': example.reshape(-1, n_features),
                                 'targets:0': target.reshape(-1, 1)}
                    _ = sess.run('train:0', feed_dict=feed_dict)

            pred = sess.run('predict:0', feed_dict={'features:0': X})
            train_errors = np.sum(pred.reshape(-1) != y)

            weights, bias = sess.run(['weights:0', 'bias:0'])

            expected_weights = np.array([[-1.2999997],
                                         [-4.0999999],
                                         [5.1999993],
                                         [2.1999998]])
            expected_bias = np.array([[-1.]])

            np.testing.assert_almost_equal(weights, expected_weights)
            np.testing.assert_almost_equal(bias, expected_bias)
            assert train_errors == 0


if __name__ == '__main__':
    unittest.main()
