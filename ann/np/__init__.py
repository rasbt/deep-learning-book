""" Submodule containing NumPy functions
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

from .preprocessing import onehot
from .preprocessing import onehot_reverse
from .preprocessing import square_padding
from .preprocessing import l2_normalize
from .scoring import accuracy_1d
from .scoring import accuracy_2d
from .training import iterate_minibatches
from .lossfunctions import mse_loss
from .lossfunctions import sse_loss
from .lossfunctions import crossentropy_loss
from .lossfunctions import log_loss
from .lossfunctions import sse_derivative
from .lossfunctions import crossentropy_derivative
from .activations import linear_activation
from .activations import linear_derivative
from .activations import logistic_activation
from .activations import logistic_derivative
from .activations import logistic_derivative_from_logistic
from .activations import tanh_activation
from .activations import tanh_derivative
from .activations import tanh_derivative_from_tanh
from .activations import relu_activation
from .activations import relu_derivative
from .activations import softplus_activation
from .activations import softplus_derivative
from .activations import softmax_activation
from .activations import softmax_derivative
