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
from ann.np import onehot


class TestOnehot(unittest.TestCase):

    def test_defaults(self):
        oh_ary = onehot(ary=np.array([0, 1, 2, 3]))
        expect = np.array([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
        self.assertTrue(np.array_equal(oh_ary, expect))
        self.assertTrue(oh_ary.dtype == np.float32)

    def test_skiplabel(self):
        oh_ary = onehot(ary=np.array([0, 1, 2, 3, 5]))
        expect = np.array([[1., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0., 0.],
                           [0., 0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 0., 1.]])
        self.assertTrue(np.array_equal(oh_ary, expect))

    def test_n_classes(self):
        oh_ary = onehot(ary=np.array([0, 1, 2]), n_classes=5)
        expect = np.array([[1., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0.],
                           [0., 0., 1., 0., 0.]])
        self.assertTrue(np.array_equal(oh_ary, expect))

    def test_dtype(self):
        oh_ary = onehot(ary=np.array([0, 1, 2]), dtype=np.int32)
        expect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        self.assertTrue(np.array_equal(oh_ary, expect))
        self.assertTrue(oh_ary.dtype == np.int32)


if __name__ == '__main__':
    unittest.main()
