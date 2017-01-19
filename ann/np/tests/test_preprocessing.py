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
from ann.np import onehot_reverse


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


class TestOnehotReverse(unittest.TestCase):

    def test_defaults(self):
        a = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 1.],
                      [0., 0., 0., 1.]])
        got = onehot_reverse(a)
        expect = np.array([0, 1, 3, 3], dtype=np.int32)
        self.assertTrue(np.array_equal(got, expect))

    def test_proba(self):
        a = np.array([[0.66, 0.24, 0.10],
                      [0.66, 0.24, 0.10],
                      [0.66, 0.24, 0.10],
                      [0.24, 0.66, 0.10]])
        got = onehot_reverse(a)
        expect = np.array([0, 0, 0, 1], dtype=np.int32)
        self.assertTrue(np.array_equal(got, expect))

    def test_dim(self):
        a = np.array([0, 0, 0, 1])
        with self.assertRaises(ValueError) as context:
            onehot_reverse(a)
        msg = context.exception
        self.assertEqual(str(msg), "Input array must have 2 dimensions\n"
                                   "Got predictions.ndim: 1")

if __name__ == '__main__':
    unittest.main()
