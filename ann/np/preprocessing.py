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


def square_padding(ary, n_elements, axes=(0, 1), value=0):
    """ Pad one or multiple arrays into square form.

    Parameters
    ----------
    ary : NumPy array, shape >= 2
        Input array consisting of 2 or more dimensions
    n_elements : int
        The number of elements in both the length and widths of the arrays
    axes : (x, y)
        The index of the x and y dimensions of the array(s) that to be padded
    value : int or float
        The value that is used to pad the array(s)

    Examples
    --------
    >>> ###################################
    >>> # pad a single 3x3 array to 5x5
    >>> ###################################
    >>> t = np.array([[1., 2., 3.],\
                      [4., 5., 6.],\
                      [7., 8., 9.]])
    >>> square_padding(ary=t, n_elements=5, axes=(0, 1))
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  2.,  3.,  0.],
           [ 0.,  4.,  5.,  6.,  0.],
           [ 0.,  7.,  8.,  9.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ###################################
    >>> # pad two 3x3 arrays to two 5x5
    >>> ###################################
    >>> t = np.array([[[1., 2., 3.],\
                       [4., 5., 6.],\
                       [7., 8., 9.]],\
                      [[10., 11., 12.],\
                       [13., 5., 6.],\
                       [7., 8., 9.]]])
    >>> square_padding(ary=t, n_elements=5, axes=(1, 2), value=0)
    array([[[  0.,   0.,   0.,   0.,   0.],
            [  0.,   1.,   2.,   3.,   0.],
            [  0.,   4.,   5.,   6.,   0.],
            [  0.,   7.,   8.,   9.,   0.],
            [  0.,   0.,   0.,   0.,   0.]],
    <BLANKLINE>
           [[  0.,   0.,   0.,   0.,   0.],
            [  0.,  10.,  11.,  12.,   0.],
            [  0.,  13.,   5.,   6.,   0.],
            [  0.,   7.,   8.,   9.,   0.],
            [  0.,   0.,   0.,   0.,   0.]]])
    """

    assert len(axes) == 2

    x_padding = n_elements - ary.shape[axes[0]]
    x_right = x_padding // 2
    x_left = x_right + (x_padding % 2)

    y_padding = n_elements - ary.shape[axes[1]]
    y_bottom = y_padding // 2
    y_top = y_bottom + (y_padding % 2)

    npad = [[0, 0] for _ in range(ary.ndim)]
    npad[axes[0]] = [x_left, x_right]
    npad[axes[1]] = [y_top, y_bottom]

    padded = np.lib.pad(array=ary,
                        pad_width=npad,
                        mode='constant',
                        constant_values=value)

    return padded


def l2_normalize(ary):
    """ Scale one or multiple vectors to unit length via L2-normalization

    Parameters
    ----------
    ary : NumPy array, shape=(,n_features) or shape=(n_vectors, n_features)
        Input array containing the vectors to be scaled. Also supports
        3D tensors of shape=(n_arrays, n_vectors, n_features).

    Examples
    --------
    >>> # 1D Tensor:
    >>> l2_normalize(np.array([1, 2, 3]))
    array([ 0.26726124,  0.53452248,  0.80178373])
    >>> # 2D Tensor:
    >>> l2_normalize(np.array([[1, 2, 3],\
                               [1, 2, 3],\
                               [3, 2, 1]]))
    array([[ 0.26726124,  0.53452248,  0.80178373],
           [ 0.26726124,  0.53452248,  0.80178373],
           [ 0.80178373,  0.53452248,  0.26726124]])
    >>> # 3D Tensor:
    >>> l2_normalize(np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],\
                                [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]))
    array([[[ 0.26726124,  0.53452248,  0.80178373],
            [ 0.26726124,  0.53452248,  0.80178373],
            [ 0.26726124,  0.53452248,  0.80178373]],
    <BLANKLINE>
           [[ 0.26726124,  0.53452248,  0.80178373],
            [ 0.26726124,  0.53452248,  0.80178373],
            [ 0.26726124,  0.53452248,  0.80178373]]])
    >>>
    """

    magnitudes = np.sqrt(np.sum(np.square(ary), axis=-1))[..., np.newaxis]
    return ary / magnitudes


if __name__ == '__main__':
    import doctest
    doctest.testmod()
