""" Activation functions implemented in NumPy
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

import numpy as np


def linear_activation(x):
    return x


def linear_derivative:
    return 1


def logistic_activation(x):
    """ Logistic Sigmoid Activation Function
    """
    return 1. / (1. + np.exp(-np.clip(x, -250, 250)))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
