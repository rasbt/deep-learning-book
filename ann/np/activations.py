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


def relu_activation(x):
    """ REctified Linear Unit activation function

    Parameters
    ----------
    x : numpy array, shape=(n_samples, )
        Input values (e.g., x.dot(weights) + bias)

    Returns
    ----------
    float
        max(0, x)

    Examples
    ----------
    >>> relu_activation(np.array([-1., 0., 2.]))
    array([-0.,  0.,  2.])
    >>>
    """
    return x * (x > 0)


def relu_derivative(x):
    """ Derivative of the REctified Linear Unit activation function

    Parameters
    ----------
    x : numpy array, shape=(n_samples, )
        Input values

    Returns
    ----------
    float
        1 if x > 0; 0, otherwise.

    Examples
    ----------
    >>> relu_derivative(np.array([-1., 0., 2.]))
    array([ 0.,  0.,  1.])
    >>>
    """
    return 1. * (x > 0)


def softplus_activation(x):
    """ Softplus activation function

    Parameters
    ----------
    x : numpy array, shape=(n_samples, )
        Input values (e.g., x.dot(weights) + bias)

    Returns
    ----------
    float
        log(1 + exp(x))

    Examples
    ----------
    >>> softplus_activation(np.array([-5., -1., 0., 2.]))
    array([ 0.00671535,  0.31326169,  0.69314718,  2.12692801])
    >>>
    """
    return np.log(1. + np.exp(x))


def softplus_derivative(x):
    """ Derivative of the softplus activation function

    Parameters
    ----------
    x : numpy array, shape=(n_samples, )
        Input values

    Returns
    ----------
    float
        logistic_sigmoid(x)

    Examples
    ----------
    >>> softplus_derivative(np.array([-1., 0., 1.]))
    array([ 0.26894142,  0.5       ,  0.73105858])
    >>>
    """
    return logistic_activation(x)


def softmax_activation(x):
    """ Softmax activation function

    Parameters
    ----------
    x : numpy array, shape=(n_samples, n_classes)
        Input values

    Returns
    ----------
    array, shape=(n_samples, n_classes)
        exp(x) / sum(exp(x))

    Examples
    ----------
    >>> softmax_activation(np.array([2.0, 1.0, 0.1]))
    array([[ 0.65900114,  0.24243297,  0.09856589]])
    >>> softmax_activation(np.array([[2.0, 1.0, 0.1],\
                                     [1.0, 2.0, 0.1],\
                                     [0.1, 1.0, 2.0],\
                                     [2.0, 1.0, 0.1]]))
    array([[ 0.65900114,  0.24243297,  0.09856589],
           [ 0.24243297,  0.65900114,  0.09856589],
           [ 0.09856589,  0.24243297,  0.65900114],
           [ 0.65900114,  0.24243297,  0.09856589]])
    """
    # np.exp(x) / np.sum(np.exp(x), axis=0)
    if x.ndim == 1:
        x = x.reshape([1, x.size])
    denom = np.exp(x - np.max(x, 1).reshape([x.shape[0], 1]))
    return denom / np.sum(denom, axis=1).reshape([denom.shape[0], 1])


def softmax_derivative(x):
    """ Derivative of the softplus activation function

    Parameters
    ----------
    x : numpy array, shape=(n_samples, n_classes)
        Input values

    Returns
    ----------
    numpy array, shape=(n_samples, n_classes)

    Examples
    ----------
    >>> softmax_derivative(np.array([[1., 2., 3.],\
                                     [4., 5., 6.]]))
    array([[ -0.08192507,  -2.18483645,  -6.22269543],
           [-12.08192507, -20.18483645, -30.22269543]])
    >>>
    """
    x_softmax = softmax_activation(x)
    jacobian = - x_softmax[:, :, np.newaxis] * x_softmax[:, np.newaxis, :]
    v_idx, h_idx = np.diag_indices(jacobian[1].shape[0])
    jacobian[:, v_idx, h_idx] = x * (1. - x)
    return jacobian.sum(axis=1)


def softmax_logloss_derivative(predictions, targets):
    """ Derivative of the softmax activation function with log loss

    Parameters
    ----------
    x : numpy array, shape=(n_samples, n_classes)
        Input values

    Returns
    ----------
    array, shape=(n_samples, n_classes)
        predictions - targets
    """
    return predictions - targets


if __name__ == '__main__':
    import doctest
    doctest.testmod()
