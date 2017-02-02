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
import numpy as np
from ann.np import mse_loss
from ann.np import sse_loss
from ann.np import sse_derivative
from ann.np import crossentropy_loss
from ann.np import crossentropy_derivative
from ann.np import log_loss


class TestMSE(unittest.TestCase):

    def test_1d(self):
        self.assertEqual(mse_loss(np.array([2.]),
                                  np.array([4.])), 4.)

    def test_2d(self):
        self.assertEqual(mse_loss(np.array([2., 3.]),
                                  np.array([4., 4.])), 2.5)


class TestSSE(unittest.TestCase):

    def test_1d(self):
        self.assertEqual(sse_loss(np.array([2.]),
                                  np.array([4.])), 4.)

    def test_2d(self):
        self.assertEqual(sse_loss(np.array([2., 3.]),
                                  np.array([4., 4.])), 5.)


class TestSSEDeriv(unittest.TestCase):

    def test_1d(self):
        self.assertEqual(sse_derivative(np.array([0.1]),
                                        np.array([0])), -0.1)

    def test_2d(self):
        self.assertEqual(sse_derivative(np.array([0.1, 2., 1.]),
                                        np.array([0, 1, 1])), -1.1)


class TestCrossEntropy(unittest.TestCase):

    def test_n_samples(self):
        softmax_out = np.array([[0.66, 0.24, 0.10],
                                [0.00, 0.77, 0.23],
                                [0.23, 0.44, 0.33],
                                [0.10, 0.24, 0.66]])

        class_labels = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]])
        self.assertEqual(round(crossentropy_loss(softmax_out,
                         class_labels), 8), 0.47834405)

    def test_1_sample(self):
        softmax_out = np.array([[0.5, 0.2, 0.3]])
        class_labels = np.array([[1.0, 0.0, 0.0]])
        self.assertEqual(round(crossentropy_loss(softmax_out,
                         class_labels), 8), 0.69314718)


class TestCrossEntDeriv(unittest.TestCase):

    def test_1d(self):
        deriv = crossentropy_derivative(np.array([0.1]),
                                        np.array([1]))
        self.assertEqual(round(deriv, 6), 0.475021)

    def test_2d(self):
        deriv = crossentropy_derivative(np.array([0.1, 2., 1.]),
                                        np.array([0, 1, 1]))
        self.assertEqual(round(deriv, 6), 0.388144)


class TestLogLoss(unittest.TestCase):
    def test_n_samples(self):
        predictions = np.array([.2, .8, .7, .3])
        class_labels = np.array([0, 1, 1, 0])
        loss = round(log_loss(predictions, class_labels), 6)
        self.assertEqual(loss, 0.289909)

if __name__ == '__main__':
    unittest.main()
