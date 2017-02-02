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


def linear_derivative(x):
    return 1


def logistic_activation(x):
    """ Logistic sigmoid activation function
    Parameters
    ----------
    x : numpy array, shape=(n_samples, )
        Input values (e.g., x.dot(weights) + bias)

    Returns
    ----------
    float
        1 / ( 1 + exp(x)

    Examples
    ----------
    >>> logistic_activation(np.array([-1, 0, 1]))
    array([ 0.26894142,  0.5       ,  0.73105858])
    >>>
    """
    return 1. / (1. + np.exp(-np.clip(x, -250, 250)))


def logistic_derivative(x):
    """ Derivative of the logistic sigmoid activation function
    Parameters
    ----------
    x : numpy array, shape=(n_samples, )
        Input values

    Returns
    ----------
    float
        logistic(x) * (1 - logistic(x))

    Examples
    ----------
    >>> logistic_derivative(np.array([-1, 0, 1]))
    array([ 0.19661193,  0.25      ,  0.19661193])
    >>>
    """
    x_logistic = logistic_activation(x)
    return x_logistic * (1. - x_logistic)


def logistic_derivative_from_logistic(x_logistic):
    """ Derivative of the logistic sigmoid activation function

    Parameters
    ----------
    x_logistic : numpy array, shape=(n_samples, )
        Output from precomputed logistic activation to save a computational
        step for efficiency.

    Returns
    ----------
    float
        x_logistic * (1 - x_logistic)

    Examples
    ----------
    >>> logistic_derivative_from_logistic(np.array([0.26894142,
    ...                                             0.5, 0.73105858]))
    array([ 0.19661193,  0.25      ,  0.19661193])
    >>>
    """
    return x_logistic * (1. - x_logistic)


def tanh_activation(x):
    """ Hyperbolic tangent (tanh sigmoid) activation function

    Parameters
    ----------
    x : numpy array, shape=(n_samples, )
        Input values (e.g., x.dot(weights) + bias)

    Returns
    ----------
    float
        (exp(x) - exp(-x)) / (e(x) + e(-x))

    Examples
    ----------
    >>> tanh_activation(np.array([-10, 0, 10]))
    array([-1.,  0.,  1.])
    >>>
    """
    return np.tanh(x)


def tanh_derivative(x):
    """ Derivative of the hyperbolic tangent (tanh sigmoid) activation function

    Parameters
    ----------
    x : numpy array, shape=(n_samples, )
        Input values

    Returns
    ----------
    float
        1 - tanh(x)**2

    Examples
    ----------
    >>> tanh_derivative(np.array([-10, 0, 10]))
    array([  8.24461455e-09,   1.00000000e+00,   8.24461455e-09])
    >>>
    """
    return 1. - tanh_activation(x)**2


def tanh_derivative_from_tanh(x_tanh):
    """ Derivative of the hyperbolic tangent (tanh sigmoid) activation function

    Parameters
    ----------
    x_tanh : numpy array, shape=(n_samples, )
        Output from precomputed tanh to save a computational
        step for efficiency.

    Returns
    ----------
    float
        1 - tanh(x)**2

    Examples
    ----------
    >>> tanh_derivative_from_tanh(np.array([-10, 0, 10]))
    array([-99.,   1., -99.])
    >>>
    """
    return 1. - x_tanh**2

if __name__ == '__main__':
    import doctest
    doctest.testmod()
