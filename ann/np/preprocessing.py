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
    >>> from ann import onehot
    >>> oh_ary = onehot(ary=np.array([0, 1, 2, 3]))
    >>> oh_ary
    array([[ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]], dtype=float32)
    >>>
    """

    if dtype is None:
        dtype = np.float32
    if n_classes is None:
        n_classes = np.max(ary) + 1
    oh_ary = (np.arange(n_classes) == ary[:, None]).astype(dtype)
    return oh_ary
