""" Perceptron model implemented in NumPy
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


def perceptron_train(features, targets, mparams=None,
                     zero_weights=True, learning_rate=1., seed=None):
    """Perceptron training function for binary class labels

    Parameters
    ----------
    features : numpy.ndarray, shape=(n_samples, m_features)
        A 2D NumPy array containing the training examples

    targets : numpy.ndarray, shape=(n_samples,)
        A 1D NumPy array containing the true class labels

    mparams : dict or None (default: None)
        A dictionary containing the model parameters, for instance
        as returned by this function. If None, a new model parameter
        dictionary is initialized. Note that the values in mparams
        are updated inplace if a mparams dict is provided.

    zero_weights : bool (default: True)
        Initializes weights to all zeros, otherwise model weights are
        initialized to small random number from a normal distribution
        with mean zero and standard deviation 0.1.

    learning_rate : float (default: 1.0)
        A learning rate for the parameter updates. Note that a learning
        rate has no effect on the direction of the decision boundary
        if if the model weights are initialized to all zeros.

    seed : int or None (default: None)
        Seed for the pseudo-random number generator that initializes the
        weights if zero_weights=False

    Returns
    -------
    mparams : dict
        The model parameters after training the perceptron for one epoch.
        The mparams dictionary has the form:
        {'weights': np.array([weight_1, weight_2, ... , weight_m]),
         'bias': np.array([bias])}

    """
    # initialize model parameters
    if mparams is None:
        mparams = {'bias': np.zeros(1)}
        if zero_weights:
            mparams['weights'] = np.zeros(features.shape[1])
        else:
            rng = np.random.RandomState(seed)
            mparams['weights'] = rng.normal(loc=0.0, scale=0.1,
                                            size=(features.shape[1]))

    # train one epoch
    for training_example, true_label in zip(features, targets):
        linear = np.dot(training_example, mparams['weights']) + mparams['bias']

        # if class 1 was predicted but true label is 0
        if linear > 0. and not true_label:
            mparams['weights'] -= learning_rate * training_example
            mparams['bias'] -= 1.

        # if class 0 was predicted but true label is 1
        elif linear <= 0. and true_label:
            mparams['weights'] += learning_rate * training_example
            mparams['bias'] += 1.

    return mparams


def perceptron_predict(features, mparams):
    """Perceptron prediction function for binary class labels

    Parameters
    ----------
    features : numpy.ndarray, shape=(n_samples, m_features)
        A 2D NumPy array containing the training examples

    mparams : dict
        The model parameters aof the perceptron in the form:
        {'weights': np.array([weight_1, weight_2, ... , weight_m]),
         'bias': np.array([bias])}

    Returns
    -------
    predicted_labels : np.ndarray, shape=(n_samples)
        NumPy array containing the predicted class labels.

    """
    linear = np.dot(features, mparams['weights']) + mparams['bias']
    predicted_labels = np.where(linear.reshape(-1) > 0., 1, 0)
    return predicted_labels


if __name__ == '__main__':
    import doctest
    doctest.testmod()
