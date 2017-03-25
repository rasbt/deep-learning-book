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


def minmax_scaling(ary, feature_minmax=(0., 1.),
                   precomputed_min=None, precomputed_max=None):
    """Rescales features to a fixed range of values

    Parameters
    ----------
    ary : 2D array, shape=(n_samples, n_features)
        The input array to be rescaled
    feature_minmax : tuple (default: (0., 1.))
        The new min and max values for each feature column.
    precomputed_min : array, shape=(n_features,) or None (default: None)
        Precomputed feature minimum values that are used for the
        rescaling if not None
    precomputed_max : array, shape=(n_features,) or None (default: None)
        Precomputed feature maximum values that are used for the
        rescaling if not None

    Returns
    -------
    rescaled_ary, precomputed_min, precomputed_max
        Returns the rescaled array, and the rescaling parameters
        for re-use.

    Examples
    --------
    >>> train_ary = np.array([[1, 1, 1],
    ...                       [4, 5, 6]])
    >>> test_ary = np.array([[1, 2, 3],
    ...                      [4, 3, 4]])
    >>> train_rescaled, tmin, tmax = minmax_scaling(train_ary,
    ...                                             feature_minmax=(0.1, 0.9))
    >>> train_rescaled
    array([[ 0.1,  0.1,  0.1],
           [ 0.9,  0.9,  0.9]])
    >>> test_rescaled, _, _ = minmax_scaling(test_ary,
    ...                                      feature_minmax=(0.1, 0.9),
    ...                                      precomputed_min=tmin,
    ...                                      precomputed_max=tmax)
    >>> test_rescaled
    array([[ 0.1 ,  0.3 ,  0.42],
           [ 0.9 ,  0.5 ,  0.58]])

    """
    if precomputed_min is None:
        precomputed_min = ary.min(axis=0)
    if precomputed_max is None:
        precomputed_max = ary.max(axis=0)

    numerator = (ary - precomputed_min) * (feature_minmax[1] -
                                           feature_minmax[0])
    denominator = (precomputed_max - precomputed_min)
    rescaled_ary = feature_minmax[0] + numerator/denominator

    return rescaled_ary, precomputed_min, precomputed_max


def standardize(ary, precomputed_mean=None, precomputed_std=None):
    """Rescales features to z-scores with zero-mean and unit variance

    Parameters
    ----------
    ary : 2D array, shape=(n_samples, n_features)
        The input array to be rescaled
    precomputed_mean : array, shape=(n_features,) or None (default: None)
        Precomputed feature mean values that are used for the
        rescaling if not None
    precomputed_std : array, shape=(n_features,) or None (default: None)
        Precomputed feature standard deviations that are used for the
        rescaling if not None

    Returns
    -------
    rescaled_ary, precomputed_mean, precomputed_std
        Returns the rescaled array, and the rescaling parameters
        for re-use.

    Examples
    --------
    >>> train_ary = np.array([[1, 1, 1],
    ...                       [4, 5, 6]])
    >>> test_ary = np.array([[1, 2, 3],
    ...                      [4, 3, 4]])
    >>> train_rescaled, tmean, tstd = standardize(train_ary)
    >>> train_rescaled
    array([[-1., -1., -1.],
           [ 1.,  1.,  1.]])
    >>> test_rescaled, _, _ = standardize(test_ary,
    ...                                   precomputed_mean=tmean,
    ...                                   precomputed_std=tstd)
    >>> test_rescaled
    array([[-1. , -0.5, -0.2],
           [ 1. ,  0. ,  0.2]])

    """

    if precomputed_std is None:
        precomputed_std = ary.std(axis=0, ddof=0)
    if precomputed_mean is None:
        precomputed_mean = ary.mean(axis=0)

    scaled_ary = (ary - precomputed_mean) / precomputed_std

    return scaled_ary, precomputed_mean, precomputed_std


def subsampling_frequent_tokens(ary, threshold=1e-5,
                                token_counts=None, seed=None):
    """ Remove frequent tokens (words) from a training corpus

    Description
    -----------
    This is an implementation of Mikolov et al's simple subsampling technique
    proposed in the improved skip-gram model for Word2Vec in
    - Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado,
      and Jeff Dean. "Distributed representations of words and
      phrases and their compositionality." In Advances in neural
      information processing systems, pp. 3111-3119. 2013.
    Using this empirical subsampling technique, each token t (or word)
    in the corpus is removed with a probability
    P(t)= 1 - sqrt(threshold/frequency(t)).

    Parameters
    ----------
    ary : array-like, shape=[n_samples, n_tokens]
        An array or list of list in which each row contains
        an arbitrary number of words or tokens.
    threshold : float (default: 1e-5)
        A positive float as subsampling threshold. The higher
        the threshold, the higher the probability of removing
        a given word.
    token_counts : dict or None (default: None)
        A dictionary with tokens as keys and token counts
        as values, which can optionally be provided to save
        computational costs if such a dictionary was
        already pre-computed.
    seed : int or None (default: None)
        A random seed for the pseudo-random number generator.

    Returns
    -------
    list, shape=(n_samples, n_tokens)
        A list of list containing the remaining, infrequent tokens that
        have not been removed from the dataset.

    Examples:
    ---------
    >>> ary = [['this', 'is', 'is', 'a', 'test'],
    ...        ['test', 'hello', 'world']]
    >>> subsampling_frequent_tokens(ary, threshold=0.1, seed=1)
    [['this', 'is', 'a'], ['hello', 'world']]
    >>> ary = ['this', 'is', 'is', 'a',
    ...        'test', 'test', 'hello', 'world']
    >>> subsampling_frequent_tokens([ary], threshold=0.1, seed=1)[0]
    ['this', 'is', 'a', 'hello', 'world']

    """
    rng = np.random.RandomState(seed)

    if token_counts is None:
        token_counts = {}
        for row in ary:
            for token in row:
                token_counts[token] = token_counts.get(token, 0) + 1

    total_count = float(sum((token_counts[k] for k in token_counts)))
    token_counts = {k: token_counts[k] / total_count for k in token_counts}

    def compute_proba(x):
        return 1 - np.sqrt(threshold / token_counts[x])

    subsampled = []
    for row in ary:
        new_row = [token for token in row
                   if compute_proba(token) < rng.rand()]
        subsampled.append(new_row)

    return subsampled


if __name__ == '__main__':
    import doctest
    doctest.testmod()
