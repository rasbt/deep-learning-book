""" Loss functions implemented in NumPy
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


def sse_loss(predictions, targets):
    """Sum squared error loss function

    Parameters
    ----------
    predictions : numpy array, shape=(n_samples, )
        Predicted values
    targets : numpy array, shape=(n_samples, )
        True target values

    Returns
    ----------
    float
        sum((predictions - targets)^2)

    Examples
    ----------
    >>> sse_loss(np.array([2., 3.]), np.array([4., 4.]))
    5.0
    >>>
    """
    return np.sum(np.square(predictions - targets))


def sse_derivative(predictions, targets):
    """Derivative of the Sum Squared error loss function

    Note that this derivative assumes the SSE form: 1/2 * SSE.
    For the "regular" SSE, use 2*sse_derivative.

    Parameters
    ----------
    predictions : numpy array, shape=(n_samples, )
        Predicted values
    targets : numpy array, shape=(n_samples, )
        True target values

    Returns
    ----------
    float
        -(predictions - targets)

    Examples
    ----------
    >>> sse_derivative(np.array([0.1, 2., 1.]), np.array([0, 1, 1]))
    -1.1000000000000001
    >>>
    """
    return np.sum(-(predictions - targets))


def mse_loss(predictions, targets):
    """Mean squared error loss function

    Parameters
    ----------
    predictions : numpy array, shape=(n_samples, )
        Predicted values
    targets : numpy array, shape=(n_samples, )
        True target values

    Returns
    ----------
    float
        sum((predictions - targets)^2) / n_samples

    Examples
    ----------
    >>> mse_loss(np.array([2., 3.]), np.array([4., 4.]))
    2.5
    >>>
    """
    return np.mean(np.square(predictions - targets))


def crossentropy_loss(softmax_predictions, onehot_targets, eps=1e-10):
    """Cross Entropy Loss Function

    Parameters
    ----------
    softmax_predictions : numpy array, shape=(n_samples, n_classes)
        Predicted values from softmax function
    onehot_targets : numpy array, shape=(n_samples, n_classes)
        True target values in one-hot encoding
    eps : float (default: 1e-10)
        Tolerance for numerical stability

    Returns
    ----------
    float
        mean[ -sum_{classes} ( target_class * log(predicted) ) ]

    Examples
    ----------
    >>> softmax_out = np.array([[0.66, 0.24, 0.10],\
                                [0.00, 0.77, 0.23],\
                                [0.23, 0.44, 0.33],\
                                [0.10, 0.24, 0.66]])

    >>> class_labels = np.array([[1.0, 0.0, 0.0],\
                                 [0.0, 1.0, 0.0],\
                                 [0.0, 1.0, 0.0],\
                                 [0.0, 0.0, 1.0]])
    >>> crossentropy_loss(softmax_out, class_labels)
    0.47834405086684895
    >>>
    """
    return np.mean(-np.sum(onehot_targets * np.log(softmax_predictions + eps),
                   axis=1))


def crossentropy_derivative(predictions, targets):
    """Derivative of the Cross Entropy loss function

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
    >>> round(crossentropy_derivative(np.array([0.1, 2., 1.]),
    ...                               np.array([0, 1, 1])), 6)
    0.38814399999999999
    >>>
    """
    return np.sum(targets / (1. + np.exp(predictions)))


def log_loss(predictions, targets, eps=1e-10):
    """Logarthmic Loss (binary cross entropy)

    Parameters
    ----------
    predictions : numpy array, shape=(n_samples)
        Predicted class probabilities in range [0, 1], where
        class 1 is the positive class
    targets : numpy array, shape=(n_samples)
        True target class labels, either 0 or 1, where 1 is the positive
        class.
    eps : float (default: 1e-10)
        Tolerance for numerical stability

    Returns
    ----------
    float
        - [ (targets * log(pred) + (1 - targets) * log(1 - pred)) ] / n_samples

    Examples
    ----------
    >>> predictions = np.array([.2, .8, .7, .3])
    >>> class_labels = np.array([0, 1, 1, 0])
    >>> log_loss(predictions, class_labels)
    0.28990924749254249
    >>>
    """
    return np.mean(- (targets * np.log(predictions + eps) +
                   (1 - targets) * np.log(1 - predictions + eps)))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
