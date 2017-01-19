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
from ann.np import accuracy_1d
from ann.np import accuracy_2d


class TestAccuracy1D(unittest.TestCase):

    def test_binary(self):
        a = np.array([1, 1, 0, 1])
        b = np.array([1, 1, 1, 1])
        self.assertEqual(accuracy_1d(a, b), 0.75)

    def test_mixed(self):
        a = np.array([1, 'a', 0, 1])
        b = np.array([1, 'b', 1, 1])
        self.assertEqual(accuracy_1d(a, b), 0.5)

    def test_unequal_len(self):
        a = np.array([1, 1, 0])
        b = np.array([1, 1, 1, 1])

        with self.assertRaises(ValueError) as context:
            accuracy_1d(a, b)
        msg = context.exception
        self.assertEqual(str(msg), "Input arrays must have same number"
                                   " of elements\n"
                                   "Got predictions.shape:"
                                   " (3,) and targets.shape: (4,)")

    def test_2d(self):
        a = np.array([[1, 1, 0, 0]])
        b = np.array([[1, 1, 1, 1]])

        with self.assertRaises(ValueError) as context:
            accuracy_1d(a, b)
        msg = context.exception
        self.assertEqual(str(msg), "Input arrays must have 1 dimension\n"
                                   "Got predictions.ndim: 2"
                                   " and targets.ndim: 2")


class TestAccuracy2D(unittest.TestCase):

    def test_binary(self):
        a = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 1.],
                      [0., 0., 0., 1.]])

        b = np.array([[0., 0., 0., 1.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 1.],
                      [0., 0., 0., 1.]])

        self.assertEqual(accuracy_2d(a, a), 1.0)
        self.assertEqual(accuracy_2d(a, b), 0.75)

    def test_unequal_len(self):
        a = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 1.]])

        b = np.array([[0., 0., 0., 1.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 1.],
                      [0., 0., 0., 1.]])
        with self.assertRaises(ValueError) as context:
            accuracy_2d(a, b)
        msg = context.exception
        self.assertEqual(str(msg), "Input arrays must have same number"
                                   " of elements\n"
                                   "Got predictions.shape:"
                                   " (3, 4) and targets.shape: (4, 4)")

    def test_minus0(self):

        a = np.array([1, 1, 0, 0])
        b = np.array([1, 1, 1, 1])

        with self.assertRaises(ValueError) as context:
            accuracy_2d(a, b)
        msg = context.exception
        self.assertEqual(str(msg), "Input arrays must have 2 dimensions\n"
                                   "Got predictions.ndim: 1"
                                   " and targets.ndim: 1")


if __name__ == '__main__':
    unittest.main()
