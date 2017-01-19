""" NumPy tools for data preprocessing
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


def onehot(ary, n_classes=None, dtype=None):
    """ One-hot encoding of NumPy arrays

    Parameters
    ----------
    ary : NumPy array, shape=(n_samples,)
        A 1D NumPy array containing class labels encoded as integers.
    n_classes : int (default: None)
        The number of class labels in `ary`. If `None` (default), the number
        of class lables is infered from the max-value in `ary`.
    dtype : NumPy dtype (default: None)
        The NumPy dtype of the one-hot encoded NumPy array that is returned.
        If dtype is `None` (default), a one-hot array, a float32 one-hot
        encoded array is returned.

    Returns
    -------
    oh_ary : int, shape=(n_samples, n_classes)
        One-hot encoded NumPy array, where sample instances are represented in
        in rows, and the number of classes is distributed across the array's
        first axis (aka columns).

    Examples
    --------
    >>> import numpy as np
    >>> from ann.np import onehot
    >>> oh_ary = onehot(ary=np.array([0, 1, 2, 3, 3]))
    >>> oh_ary
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  1.]], dtype=float32)
    >>>
    """

    if dtype is None:
        dtype = np.float32
    if n_classes is None:
        n_classes = np.max(ary) + 1
    oh_ary = (np.arange(n_classes) == ary[:, None]).astype(dtype)
    return oh_ary


def onehot_reverse(predictions, dtype=None):
    """ Turns one-hot arrays or class probabilities back into class labels

    Parameters
    ----------
    ary : NumPy array, shape=(n_samples, n_classes)
        A 2D NumPy array in onehot format or class probabilities
    dtype : NumPy dtype (default: None)
        The NumPy dtype of the 1D NumPy array that is returned.
        If dtype is `None` (default), a one-hot array, returns an int32 array

    Returns
    -------
    array-like, shape=(n_classes)
        Class label array

    Examples
    --------
    >>> import numpy as np
    >>> from ann.np import onehot_reverse
    >>> a = np.array([[ 1.,  0.,  0.,  0.],\
                      [ 0.,  1.,  0.,  0.],\
                      [ 0.,  0.,  0.,  1.],\
                      [ 0.,  0.,  0.,  1.]])
    >>> onehot_reverse(a)
    array([0, 1, 3, 3], dtype=int32)
    >>> b = np.array([[0.66, 0.24, 0.10],\
                      [0.66, 0.24, 0.10],\
                      [0.66, 0.24, 0.10],\
                      [0.24, 0.66, 0.10]])
    >>> onehot_reverse(b)
    array([0, 0, 0, 1], dtype=int32)
    >>>
    """
    if dtype is None:
        dtype = np.int32
    if predictions.ndim != 2:
        raise ValueError("Input array must have 2 dimensions\n"
                         "Got predictions.ndim: %d" % predictions.ndim)
    return np.argmax(predictions, axis=1).astype(dtype)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
