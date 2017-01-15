# Sebastian Raschka 2016-2017
#
# ann is a supporting package for the book
# "Introduction to Artificial Neural Networks and Deep Learning:
#  A Practical Guide with Applications in Python"
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import numpy as np
from ann import onehot


def test_default():
    oh_ary = onehot(ary=np.array([0, 1, 2, 3]))
    expect = np.array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])
    assert np.array_equal(oh_ary, expect)
    assert oh_ary.dtype == np.float32


def test_skiplabel():
    oh_ary = onehot(ary=np.array([0, 1, 2, 3, 5]))
    expect = np.array([[1., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 0., 1.]])
    assert np.array_equal(oh_ary, expect)


def test_n_classes():
    oh_ary = onehot(ary=np.array([0, 1, 2]), n_classes=5)
    expect = np.array([[1., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0.],
                       [0., 0., 1., 0., 0.]])
    assert np.array_equal(oh_ary, expect)


def test_dtype():
    oh_ary = onehot(ary=np.array([0, 1, 2]), dtype=np.int32)
    expect = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
    assert np.array_equal(oh_ary, expect)
    assert oh_ary.dtype == np.int32
