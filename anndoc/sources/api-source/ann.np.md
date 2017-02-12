ann version: 0.1.0
## accuracy_1d

*accuracy_1d(predictions, targets)*

Computes the prediction accuracy from class labels in 1D NumPy arrays.

**Parameters**

- `predictions` : array_like, shape=(n_samples,)

    1-dimensional NumPy array containing predicted class labels.

- `targets` : array_like, shape=(n_samples,)

    1-dimensional NumPy array containing the true class labels.

**Returns**

float
    The prediction accuracy (fraction of samples that was predicted
    correctly) in the range [0, 1], where 1 is best.

**Examples**

    >>> import numpy as np
    >>> from ann.np import accuracy_1d
    >>> a = np.array([1, 1, 0, 1])
    >>> b = np.array([1, 1, 1, 1])
    >>> accuracy_1d(a, b)
    0.75
    >>>

## accuracy_2d

*accuracy_2d(predictions, targets)*

Computes the prediction accuracy from class labels in onehot
    encoded 2D NumPy arrays.

**Parameters**

- `predictions` : array_like, shape=(n_samples, n_classes)

    2-dimensional NumPy array in onehot-encoded format.

- `targets` : array_like, shape=(n_samples, n_classes)

    2-dimensional NumPy array in onehot-encoded format.

**Returns**

float
    The prediction accuracy (fraction of samples that was predicted
    correctly) in the range [0, 1], where 1 is best.

**Examples**

    >>> import numpy as np
    >>> from ann.np import accuracy_2d
    >>> a = np.array([[ 1.,  0.,  0.,  0.],                      [ 0.,  1.,  0.,  0.],                      [ 0.,  0.,  0.,  1.],                      [ 0.,  0.,  0.,  1.]])
    >>> accuracy_2d(a, a)
    1.0
    >>> b = np.array([[ 0.,  0.,  0.,  1.],                      [ 0.,  1.,  0.,  0.],                      [ 0.,  0.,  0.,  1.],                      [ 0.,  0.,  0.,  1.]])
    >>> accuracy_2d(a, b)
    0.75
    >>>

## crossentropy_derivative

*crossentropy_derivative(predictions, targets)*

Derivative of the Cross Entropy loss function

**Parameters**

- `predictions` : numpy array, shape=(n_samples, )

    Predicted values

- `targets` : numpy array, shape=(n_samples, )

    True target values

**Returns**

float
    sum[ targets / (1 + exp(predictions)) ]

**Examples**

    >>> round(crossentropy_derivative(np.array([0.1, 2., 1.]),
    ...                               np.array([0, 1, 1])), 6)
    bla
    0.38814399999999999
    >>>

## crossentropy_loss

*crossentropy_loss(softmax_predictions, onehot_targets, eps=1e-10)*

Cross Entropy Loss Function

**Parameters**

- `softmax_predictions` : numpy array, shape=(n_samples, n_classes)

    Predicted values from softmax function

- `onehot_targets` : numpy array, shape=(n_samples, n_classes)

    True target values in one-hot encoding

- `eps` : float (default: 1e-10)

    Tolerance for numerical stability

**Returns**

float
    mean[ -sum_{classes} ( target_class * log(predicted) ) ]

**Examples**

    >>> softmax_out = np.array([[0.66, 0.24, 0.10],                                [0.00, 0.77, 0.23],                                [0.23, 0.44, 0.33],                                [0.10, 0.24, 0.66]])
    >>> class_labels = np.array([[1.0, 0.0, 0.0],                                 [0.0, 1.0, 0.0],                                 [0.0, 1.0, 0.0],                                 [0.0, 0.0, 1.0]])
    >>> crossentropy_loss(softmax_out, class_labels)
    0.47834405086684895
    >>>

## iterate_minibatches

*iterate_minibatches(arrays, batch_size, shuffle=False, seed=None)*

Yields minibatches over one epoch.

**Parameters**

- `data` : iterable

    An iterable of arrays, where the first axis of each array goes
    over the number of samples.

- `batch_size` : NumPy dtype (default: None)

    The NumPy dtype of the one-hot encoded NumPy array that is returned.
    If dtype is `None` (default), a one-hot array, a float32 one-hot
    encoded array is returned.
    Note that if the array length is not divisible by the batch size,
    the remaining sample instances are not included in the last batch.
    This is to guarantee similar-sized batches.

- `shuffle` : Bool (default: False)

    Minibatches are returned from shuffled arrays if `True`.
    Arrays are shuffled in unison, i.e., their relative order is
    maintained. Also, the original arrays are not being modified in place.)

- `seed` : int or None (default: None)

    Uses a random seed for shuffling if `seed` is not `None`
    This parameter has no effect if shuffle=`False`.

**Yields**

generator
    A generator object containing a minibatch from
    each array, i.e., (array0_minibatch, array1_minibatch, ...)

**Examples**

    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5, 6])
    >>> y = np.array([7, 8, 9, 10, 11, 12])
    >>> mb1 = iterate_minibatches(arrays=(x, y), batch_size=2)
    >>> for x_batch, y_batch in mb1:
    ...     print(x_batch, y_batch)
    bla
    [1 2] [7 8]
    [3 4] [ 9 10]
    [5 6] [11 12]
    >>> mb2 = iterate_minibatches(arrays=(x, y), batch_size=2,                                  shuffle=True, seed=123)
    >>> for x_batch, y_batch in mb2:
    ...     print(x_batch, y_batch)
    bla
    [2 4] [ 8 10]
    [5 1] [11  7]
    [3 6] [ 9 12]
    >>> # Note that if the array length is not divisible by the batch size
    >>> #
    >>> #
    >>> mb3 = iterate_minibatches(arrays=(x, y), batch_size=4)
    >>> for x_batch, y_batch in mb3:
    ...     print(x_batch, y_batch)
    bla
    [1 2 3 4] [ 7  8  9 10]
    >>>

## linear_activation

*linear_activation(x)*

None

## linear_derivative

*linear_derivative(x)*

None

## log_loss

*log_loss(predictions, targets, eps=1e-10)*

Logarthmic Loss (binary cross entropy)

**Parameters**

- `predictions` : numpy array, shape=(n_samples)

    Predicted class probabilities in range [0, 1], where
    class 1 is the positive class

- `targets` : numpy array, shape=(n_samples)

    True target class labels, either 0 or 1, where 1 is the positive
    class.

- `eps` : float (default: 1e-10)

    Tolerance for numerical stability

**Returns**

float
    - [ (targets * log(pred) + (1 - targets) * log(1 - pred)) ] / n_samples

**Examples**

    >>> predictions = np.array([.2, .8, .7, .3])
    >>> class_labels = np.array([0, 1, 1, 0])
    >>> log_loss(predictions, class_labels)
    0.28990924749254249
    >>>

## logistic_activation

*logistic_activation(x)*

Logistic sigmoid activation function
**Parameters**

- `x` : numpy array, shape=(n_samples, )

    Input values (e.g., x.dot(weights) + bias)

**Returns**

float
    1 / ( 1 + exp(x)

**Examples**

    >>> logistic_activation(np.array([-1, 0, 1]))
    array([ 0.26894142,  0.5       ,  0.73105858])
    >>>

## logistic_derivative

*logistic_derivative(x)*

Derivative of the logistic sigmoid activation function
**Parameters**

- `x` : numpy array, shape=(n_samples, )

    Input values

**Returns**

float
    logistic(x) * (1 - logistic(x))

**Examples**

    >>> logistic_derivative(np.array([-1, 0, 1]))
    array([ 0.19661193,  0.25      ,  0.19661193])
    >>>

## logistic_derivative_from_logistic

*logistic_derivative_from_logistic(x_logistic)*

Derivative of the logistic sigmoid activation function

**Parameters**

- `x_logistic` : numpy array, shape=(n_samples, )

    Output from precomputed logistic activation to save a computational
    step for efficiency.

**Returns**

float
    x_logistic * (1 - x_logistic)

**Examples**

    >>> logistic_derivative_from_logistic(np.array([0.26894142,
    ...                                             0.5, 0.73105858]))
    bla
    array([ 0.19661193,  0.25      ,  0.19661193])
    >>>

## mse_loss

*mse_loss(predictions, targets)*

Mean squared error loss function

**Parameters**

- `predictions` : numpy array, shape=(n_samples, )

    Predicted values

- `targets` : numpy array, shape=(n_samples, )

    True target values

**Returns**

float
    sum((predictions - targets)^2) / n_samples

**Examples**

    >>> mse_loss(np.array([2., 3.]), np.array([4., 4.]))
    2.5
    >>>

## onehot

*onehot(ary, n_classes=None, dtype=None)*

One-hot encoding of NumPy arrays

**Parameters**

- `ary` : NumPy array, shape=(n_samples,)

    A 1D NumPy array containing class labels encoded as integers.

- `n_classes` : int (default: None)

    The number of class labels in `ary`. If `None` (default), the number
    of class lables is infered from the max-value in `ary`.

- `dtype` : NumPy dtype (default: None)

    The NumPy dtype of the one-hot encoded NumPy array that is returned.
    If dtype is `None` (default), a one-hot array, a float32 one-hot
    encoded array is returned.

**Returns**

- `oh_ary` : int, shape=(n_samples, n_classes)

    One-hot encoded NumPy array, where sample instances are represented in
    in rows, and the number of classes is distributed across the array's
    first axis (aka columns).

**Examples**

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

## onehot_reverse

*onehot_reverse(predictions, dtype=None)*

Turns one-hot arrays or class probabilities back into class labels

**Parameters**

- `ary` : NumPy array, shape=(n_samples, n_classes)

    A 2D NumPy array in onehot format or class probabilities

- `dtype` : NumPy dtype (default: None)

    The NumPy dtype of the 1D NumPy array that is returned.
    If dtype is `None` (default), a one-hot array, returns an int32 array

**Returns**

array-like, shape=(n_classes)
    Class label array

**Examples**

    >>> import numpy as np
    >>> from ann.np import onehot_reverse
    >>> a = np.array([[ 1.,  0.,  0.,  0.],                      [ 0.,  1.,  0.,  0.],                      [ 0.,  0.,  0.,  1.],                      [ 0.,  0.,  0.,  1.]])
    >>> onehot_reverse(a)
    array([0, 1, 3, 3], dtype=int32)
    >>> b = np.array([[0.66, 0.24, 0.10],                      [0.66, 0.24, 0.10],                      [0.66, 0.24, 0.10],                      [0.24, 0.66, 0.10]])
    >>> onehot_reverse(b)
    array([0, 0, 0, 1], dtype=int32)
    >>>

## relu_activation

*relu_activation(x)*

REctified Linear Unit activation function

**Parameters**

- `x` : numpy array, shape=(n_samples, )

    Input values (e.g., x.dot(weights) + bias)

**Returns**

float
    max(0, x)

**Examples**

    >>> relu_activation(np.array([-1., 0., 2.]))
    array([-0.,  0.,  2.])
    >>>

## relu_derivative

*relu_derivative(x)*

Derivative of the REctified Linear Unit activation function

**Parameters**

- `x` : numpy array, shape=(n_samples, )

    Input values

**Returns**

float
    1 if x > 0; 0, otherwise.

**Examples**

    >>> relu_derivative(np.array([-1., 0., 2.]))
    array([ 0.,  0.,  1.])
    >>>

## softmax_activation

*softmax_activation(x)*

Softmax activation function

**Parameters**

- `x` : numpy array, shape=(n_samples, n_classes)

    Input values

**Returns**

array, shape=(n_samples, n_classes)
    exp(x) / sum(exp(x))

**Examples**

    >>> softmax_activation(np.array([2.0, 1.0, 0.1]))
    array([[ 0.65900114,  0.24243297,  0.09856589]])
    >>> softmax_activation(np.array([[2.0, 1.0, 0.1],                                     [1.0, 2.0, 0.1],                                     [0.1, 1.0, 2.0],                                     [2.0, 1.0, 0.1]]))
    array([[ 0.65900114,  0.24243297,  0.09856589],
    [ 0.24243297,  0.65900114,  0.09856589],
    [ 0.09856589,  0.24243297,  0.65900114],
    [ 0.65900114,  0.24243297,  0.09856589]])

## softmax_derivative

*softmax_derivative(x)*

Derivative of the softplus activation function

**Parameters**

- `x` : numpy array, shape=(n_samples, n_classes)

    Input values

**Returns**

numpy array, shape=(n_samples, n_classes)

**Examples**

    >>> softmax_derivative(np.array([[1., 2., 3.],                                     [4., 5., 6.]]))
    array([[ -0.08192507,  -2.18483645,  -6.22269543],
    [-12.08192507, -20.18483645, -30.22269543]])
    >>>

## softplus_activation

*softplus_activation(x)*

Softplus activation function

**Parameters**

- `x` : numpy array, shape=(n_samples, )

    Input values (e.g., x.dot(weights) + bias)

**Returns**

float
    log(1 + exp(x))

**Examples**

    >>> softplus_activation(np.array([-5., -1., 0., 2.]))
    array([ 0.00671535,  0.31326169,  0.69314718,  2.12692801])
    >>>

## softplus_derivative

*softplus_derivative(x)*

Derivative of the softplus activation function

**Parameters**

- `x` : numpy array, shape=(n_samples, )

    Input values

**Returns**

float
    logistic_sigmoid(x)

**Examples**

    >>> softplus_derivative(np.array([-1., 0., 1.]))
    array([ 0.26894142,  0.5       ,  0.73105858])
    >>>

## square_padding

*square_padding(ary, n_elements, axes=(0, 1), value=0)*

Pad one or multiple arrays into square form.

**Parameters**

- `ary` : NumPy array, shape >= 2

    Input array consisting of 2 or more dimensions

- `n_elements` : int

    The number of elements in both the length and widths of the arrays

- `axes` : (x, y)

    The index of the x and y dimensions of the array(s) that to be padded

- `value` : int or float

    The value that is used to pad the array(s)

**Examples**

    >>> ###################################
    >>> # pad a single 3x3 array to 5x5
    >>> ###################################
    >>> t = np.array([[1., 2., 3.],                      [4., 5., 6.],                      [7., 8., 9.]])
    >>> square_padding(ary=t, n_elements=5, axes=(0, 1))
    array([[ 0.,  0.,  0.,  0.,  0.],
    [ 0.,  1.,  2.,  3.,  0.],
    [ 0.,  4.,  5.,  6.,  0.],
    [ 0.,  7.,  8.,  9.,  0.],
    [ 0.,  0.,  0.,  0.,  0.]])
    >>> ###################################
    >>> # pad two 3x3 arrays to two 5x5
    >>> ###################################
    >>> t = np.array([[[1., 2., 3.],                       [4., 5., 6.],                       [7., 8., 9.]],                      [[10., 11., 12.],                       [13., 5., 6.],                       [7., 8., 9.]]])
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

## sse_derivative

*sse_derivative(predictions, targets)*

Derivative of the Sum Squared error loss function

    Note that this derivative assumes the SSE form: 1/2 * SSE.
    For the "regular" SSE, use 2*sse_derivative.

**Parameters**

- `predictions` : numpy array, shape=(n_samples, )

    Predicted values

- `targets` : numpy array, shape=(n_samples, )

    True target values

**Returns**

float
    -(predictions - targets)

**Examples**

    >>> sse_derivative(np.array([0.1, 2., 1.]), np.array([0, 1, 1]))
    -1.1000000000000001
    >>>

## sse_loss

*sse_loss(predictions, targets)*

Sum squared error loss function

**Parameters**

- `predictions` : numpy array, shape=(n_samples, )

    Predicted values

- `targets` : numpy array, shape=(n_samples, )

    True target values

**Returns**

float
    sum((predictions - targets)^2)

**Examples**

    >>> sse_loss(np.array([2., 3.]), np.array([4., 4.]))
    5.0
    >>>

## tanh_activation

*tanh_activation(x)*

Hyperbolic tangent (tanh sigmoid) activation function

**Parameters**

- `x` : numpy array, shape=(n_samples, )

    Input values (e.g., x.dot(weights) + bias)

**Returns**

float
    (exp(x) - exp(-x)) / (e(x) + e(-x))

**Examples**

    >>> tanh_activation(np.array([-10, 0, 10]))
    array([-1.,  0.,  1.])
    >>>

## tanh_derivative

*tanh_derivative(x)*

Derivative of the hyperbolic tangent (tanh sigmoid) activation function

**Parameters**

- `x` : numpy array, shape=(n_samples, )

    Input values

**Returns**

float
1 - tanh(x)**2

**Examples**

    >>> tanh_derivative(np.array([-10, 0, 10]))
    array([  8.24461455e-09,   1.00000000e+00,   8.24461455e-09])
    >>>

## tanh_derivative_from_tanh

*tanh_derivative_from_tanh(x_tanh)*

Derivative of the hyperbolic tangent (tanh sigmoid) activation function

**Parameters**

- `x_tanh` : numpy array, shape=(n_samples, )

    Output from precomputed tanh to save a computational
    step for efficiency.

**Returns**

float
1 - tanh(x)**2

**Examples**

    >>> tanh_derivative_from_tanh(np.array([-10, 0, 10]))
    array([-99.,   1., -99.])
    >>>

