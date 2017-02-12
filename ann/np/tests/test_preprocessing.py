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
from ann.np import square_padding
from ann.np import l2_normalize


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


class TestSquarePadding(unittest.TestCase):
    def test_1ary_defaults_3to7(self):
        a = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        exp = np.array([[0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 2., 3., 0., 0.],
                        [0., 0., 4., 5., 6., 0., 0.],
                        [0., 0., 7., 8., 9., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0.]])
        got = square_padding(ary=a, n_elements=7, value=0)
        np.testing.assert_allclose(exp, got)

    def test_1ary_defaults_3to6(self):
        a = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        exp = np.array([[0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 2., 3., 0.],
                        [0., 0., 4., 5., 6., 0.],
                        [0., 0., 7., 8., 9., 0.],
                        [0., 0., 0., 0., 0., 0.]])
        got = square_padding(ary=a, n_elements=6, value=0)
        np.testing.assert_allclose(exp, got)

    def test_2ary_defaults_3to7(self):
        a = np.array([[[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.]],
                      [[10., 11., 12.],
                       [13., 5., 6.],
                       [7., 8., 9.]]])

        exp = np.array([[[0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 2., 3., 0., 0.],
                         [0., 0., 4., 5., 6., 0., 0.],
                         [0., 0., 7., 8., 9., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.]],
                        [[0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 10., 11., 12., 0., 0.],
                         [0., 0., 13., 5., 6., 0., 0.],
                         [0., 0., 7., 8., 9., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.]]])

        got = square_padding(ary=a, n_elements=7, axes=(1, 2), value=0)
        np.testing.assert_allclose(exp, got)

    def test_2ary_defaults_3to6(self):
        a = np.array([[[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.]],
                      [[10., 11., 12.],
                       [13., 5., 6.],
                       [7., 8., 9.]]])

        got = square_padding(ary=a, n_elements=6, axes=(1, 2), value=0)

        exp = np.array([[[0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 2., 3., 0.],
                         [0., 0., 4., 5., 6., 0.],
                         [0., 0., 7., 8., 9., 0.],
                         [0., 0., 0., 0., 0., 0.]],
                        [[0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0.],
                         [0., 0., 10., 11., 12., 0.],
                         [0., 0., 13., 5., 6., 0.],
                         [0., 0., 7., 8., 9., 0.],
                         [0., 0., 0., 0., 0., 0.]]])
        np.testing.assert_allclose(exp, got)


class TestL2Normalize(unittest.TestCase):
    def test_1d(self):
        exp = np.array([0.26726124, 0.53452248, 0.80178373])
        got = l2_normalize(np.array([1, 2, 3]))
        np.testing.assert_allclose(exp, got)

    def test_2d(self):
        exp = np.array([[0.26726124, 0.53452248, 0.80178373],
                        [0.26726124, 0.53452248, 0.80178373],
                        [0.80178373, 0.53452248, 0.26726124]])
        got = l2_normalize(np.array([[1, 2, 3], [1, 2, 3], [3, 2, 1]]))
        np.testing.assert_allclose(exp, got)

    def test_3d(self):
        exp = np.array([[[0.26726124, 0.53452248, 0.80178373],
                         [0.26726124, 0.53452248, 0.80178373],
                         [0.26726124, 0.53452248, 0.80178373]],
                        [[0.26726124, 0.53452248, 0.80178373],
                         [0.26726124, 0.53452248, 0.80178373],
                         [0.26726124, 0.53452248, 0.80178373]]])
        got = l2_normalize(np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                                     [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]))
        np.testing.assert_allclose(exp, got)


if __name__ == '__main__':
    unittest.main()
