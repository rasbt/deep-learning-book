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
    Parameters
    ----------
    predictions : numpy array, shape=(n_samples, )
        Predicted values
    targets : numpy array, shape=(n_samples, )
        True target values

    Returns
    ----------
    float
        sum[ targets / (1 + exp(predictions)) ]

    Examples
    ----------
    >>> round(crossentropy_derivative(np.array([0.1, 2., 1.]), \
    ...                               np.array([0, 1, 1])), 6)
    0.388144
    >>>
    """
    return 1. / (1. + np.exp(-np.clip(x, -250, 250)))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
