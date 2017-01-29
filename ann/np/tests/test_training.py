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
from ann.np import iterate_minibatches


class TestIterateMinibatches(unittest.TestCase):

    def test_default(self):
        a = np.array([1, 2, 3, 4, 5, 6])
        b = np.array([7, 8, 9, 10, 11, 12])
        expect1 = np.array([[1, 2], [3, 4], [5, 6]])
        expect2 = np.array([[7, 8], [9, 10], [11, 12]])
        for idx, m in enumerate(iterate_minibatches((a, b), batch_size=2)):
            m = list(m)
            self.assertTrue(np.array_equal(m[0], expect1[idx]))
            self.assertTrue(np.array_equal(m[1], expect2[idx]))

    def test_default_uneven(self):
        a = np.array([1, 2, 3, 4, 5, 6])
        b = np.array([7, 8, 9, 10, 11, 12])
        expect1 = np.array([[1, 2, 3, 4]])
        expect2 = np.array([[7, 8, 9, 10]])
        for idx, m in enumerate(iterate_minibatches((a, b), batch_size=4)):
            m = list(m)
            self.assertTrue(np.array_equal(m[0], expect1[idx]))
            self.assertTrue(np.array_equal(m[1], expect2[idx]))

    def test_default_maxbatch(self):
        a = np.array([1, 2, 3, 4, 5, 6])
        b = np.array([7, 8, 9, 10, 11, 12])
        it = iterate_minibatches((a, b), batch_size=6)
        m = list(next(it))
        self.assertTrue(np.array_equal(m[0], a))
        self.assertTrue(np.array_equal(m[1], b))

    def test_default_hugebatch(self):
        a = np.array([1, 2, 3, 4, 5, 6])
        b = np.array([7, 8, 9, 10, 11, 12])
        it = iterate_minibatches((a, b), batch_size=99)
        with self.assertRaises(StopIteration):
            m = next(it)

    def test_shuffle(self):
        a = np.array([1, 2, 3, 4, 5, 6])
        b = np.array([7, 8, 9, 10, 11, 12])
        expect1 = np.array([[2, 4], [5, 1], [3, 6]])
        expect2 = np.array([[8, 10], [11, 7], [9, 12]])
        for idx, m in enumerate(iterate_minibatches((a, b), batch_size=2,
                                shuffle=True, seed=123)):
            m = list(m)
            self.assertTrue(np.array_equal(m[0], expect1[idx]))
            self.assertTrue(np.array_equal(m[1], expect2[idx]))


if __name__ == '__main__':
    unittest.main()
