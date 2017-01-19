""" NumPy tools for scoring and evaluation
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


def accuracy_1d(predictions, targets):
    """Computes the prediction accuracy from class labels in 1D NumPy arrays.

    Parameters
    ----------
    predictions : array_like, shape=(n_samples,)
        1-dimensional NumPy array containing predicted class labels.
    targets : array_like, shape=(n_samples,)
        1-dimensional NumPy array containing the true class labels.

    Returns
    -------
    float
        The prediction accuracy (fraction of samples that was predicted
        correctly) in the range [0, 1], where 1 is best.

    Examples
    --------
    >>> import numpy as np
    >>> from ann.np import accuracy_1d
    >>> a = np.array([1, 1, 0, 1])
    >>> b = np.array([1, 1, 1, 1])
    >>> accuracy_1d(a, b)
    0.75
    >>>
    """

    if predictions.ndim > 1 or targets.ndim > 1:
        raise ValueError("Input arrays must have 1 dimension\n"
                         "Got predictions.ndim: %d and targets.ndim: %d"
                         % (predictions.ndim, targets.ndim))

    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("Input arrays must have same number of elements\n"
                         "Got predictions.shape: %s and targets.shape: %s"
                         % (predictions.shape, targets.shape))

    return np.sum(predictions == targets) / float(predictions.shape[0])


def accuracy_2d(predictions, targets):
    """Computes the prediction accuracy from class labels in onehot
    encoded 2D NumPy arrays.

    Parameters
    ----------
    predictions : array_like, shape=(n_samples, n_classes)
        2-dimensional NumPy array in onehot-encoded format.
    targets : array_like, shape=(n_samples, n_classes)
        2-dimensional NumPy array in onehot-encoded format.

    Returns
    -------
    float
        The prediction accuracy (fraction of samples that was predicted
        correctly) in the range [0, 1], where 1 is best.

    Examples
    --------
    >>> import numpy as np
    >>> from ann.np import accuracy_2d
    >>> a = np.array([[ 1.,  0.,  0.,  0.],\
                      [ 0.,  1.,  0.,  0.],\
                      [ 0.,  0.,  0.,  1.],\
                      [ 0.,  0.,  0.,  1.]])
    >>> accuracy_2d(a, a)
    1.0
    >>> b = np.array([[ 0.,  0.,  0.,  1.],\
                      [ 0.,  1.,  0.,  0.],\
                      [ 0.,  0.,  0.,  1.],\
                      [ 0.,  0.,  0.,  1.]])
    >>> accuracy_2d(a, b)
    0.75
    >>>
    """

    if predictions.ndim != 2 or targets.ndim != 2:
        raise ValueError("Input arrays must have 2 dimensions\n"
                         "Got predictions.ndim: %d and targets.ndim: %d"
                         % (predictions.ndim, targets.ndim))

    if predictions.shape != targets.shape:
        raise ValueError("Input arrays must have same number of elements\n"
                         "Got predictions.shape: %s and targets.shape: %s"
                         % (predictions.shape, targets.shape))

    p_flat = np.argmax(predictions, axis=1)
    t_flat = np.argmax(targets, axis=1)
    return np.sum(p_flat == t_flat) / float(predictions.shape[0])


if __name__ == '__main__':
    import doctest
    doctest.testmod()
