""" NumPy tools for training
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


def iterate_minibatches(arrays, batch_size, shuffle=False, seed=None):
    """Yields minibatches over one epoch.

    Parameters
    ----------
    data : iterable
        An iterable of arrays, where the first axis of each array goes
        over the number of samples.
    batch_size : NumPy dtype (default: None)
        The NumPy dtype of the one-hot encoded NumPy array that is returned.
        If dtype is `None` (default), a one-hot array, a float32 one-hot
        encoded array is returned.
        Note that if the array length is not divisible by the batch size,
        the remaining sample instances are not included in the last batch.
        This is to guarantee similar-sized batches.
    shuffle : Bool (default: False)
        Minibatches are returned from shuffled arrays if `True`.
        Arrays are shuffled in unison, i.e., their relative order is
        maintained. Also, the original arrays are not being modified in place.)
    seed : int or None (default: None)
        Uses a random seed for shuffling if `seed` is not `None`
        This parameter has no effect if shuffle=`False`.

    Yields
    -------
    generator
        A generator object containing a minibatch from
        each array, i.e., (array0_minibatch, array1_minibatch, ...)

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5, 6])
    >>> y = np.array([7, 8, 9, 10, 11, 12])
    >>> mb1 = iterate_minibatches(arrays=(x, y), batch_size=2)
    >>> for x_batch, y_batch in mb1:
    ...     print(x_batch, y_batch)
    [1 2] [7 8]
    [3 4] [ 9 10]
    [5 6] [11 12]
    >>> mb2 = iterate_minibatches(arrays=(x, y), batch_size=2,\
                                  shuffle=True, seed=123)
    >>> for x_batch, y_batch in mb2:
    ...     print(x_batch, y_batch)
    [2 4] [ 8 10]
    [5 1] [11  7]
    [3 6] [ 9 12]
    >>> # Note that if the array length is not divisible by the batch size
    >>> #
    >>> #
    >>> mb3 = iterate_minibatches(arrays=(x, y), batch_size=4)
    >>> for x_batch, y_batch in mb3:
    ...     print(x_batch, y_batch)
    [1 2 3 4] [ 7  8  9 10]
    >>>
    """
    rgen = np.random.RandomState(seed)
    indices = np.arange(arrays[0].shape[0])

    if shuffle:
        rgen.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
        index_slice = indices[start_idx:start_idx + batch_size]

        yield (ary[index_slice] for ary in arrays)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
