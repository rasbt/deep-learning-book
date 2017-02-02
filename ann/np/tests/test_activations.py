# Sebastian Raschka 2016-2017
#
# ann is a supporting package for the book
# "Introduction to Artificial Neural Networks and Deep Learning:
#  A Practical Guide with Applications in Python"
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import unittest
import numpy as np
from ann.np import logistic_activation
from ann.np import logistic_derivative
from ann.np import logistic_derivative_from_logistic
from ann.np import tanh_activation
from ann.np import tanh_derivative
from ann.np import tanh_derivative_from_tanh
from ann.np import relu_activation
from ann.np import relu_derivative
from ann.np import softplus_activation
from ann.np import softplus_derivative
from ann.np import softmax_activation
from ann.np import softmax_derivative
from ann.np import softmax_derivative


class TestLogActivation(unittest.TestCase):

    def test_n_samples(self):
        self.assertTrue(np.allclose(
                        logistic_activation(np.array([-1, 0, 1])),
                        np.array([0.268941, 0.5, 0.731059])))

    def test_one_sample(self):
        self.assertTrue(np.allclose(
                        logistic_activation(np.array([-1])),
                        np.array([0.268941])))


class TestLogActivationDerivative(unittest.TestCase):

    def test_n_samples_deriv(self):

        self.assertTrue(np.allclose(
                        logistic_derivative(np.array([-1., 0., 1.])),
                        np.array([0.196612, 0.25, 0.196612])))

    def test_one_sample_deriv(self):
        self.assertTrue(np.allclose(
                        logistic_derivative(np.array([-1])),
                        np.array([0.196612])))

    def test_n_samples_deriv_from_logistic(self):

        sig = np.array([0.26894142, 0.5, 0.73105858])
        self.assertTrue(np.allclose(logistic_derivative_from_logistic(sig),
                        np.array([0.196612, 0.25, 0.196612])))

    def test_one_sample_deriv_from_logistic(self):
        self.assertTrue(np.allclose(
            logistic_derivative_from_logistic(np.array([0.26894142])),
            np.array([0.196612])))


class TestTanh(unittest.TestCase):

    def test_one_sample(self):
        def manual(x):
            a, b = np.exp(x), np.exp(-x)
            return (a - b) / (a + b)
        self.assertTrue(np.allclose(
                        tanh_activation(np.array([1])),
                        manual(np.array([1]))))


class TestTanhDerivative(unittest.TestCase):

    def test_n_samples_deriv(self):

        self.assertTrue(np.allclose(
                        tanh_derivative(np.array([0.2689414, 0.5, 0.7310586])),
                        np.array([0.93102047, 0.78644773, 0.61098265])))

    def test_one_sample_deriv(self):
        self.assertTrue(np.allclose(
                        tanh_derivative(np.array([0.2689414])),
                        np.array([0.93102047])))

    def test_n_samples_deriv_from_tanh(self):
        tan = np.array([0.26263955, 0.46211716, 0.62371255])
        self.assertTrue(np.allclose(tanh_derivative_from_tanh(tan),
                        np.array([0.93102047, 0.78644773, 0.61098265])))

    def test_one_sample_deriv_from_tanh(self):
        self.assertTrue(np.allclose(
            tanh_derivative_from_tanh(np.array([0.26263955])),
            np.array([0.93102047])))


class TestRelu(unittest.TestCase):

    def test_n_samples(self):
        x = relu_activation(np.array([-1., 0., 2.]))
        act = np.array([0., 0., 2.])
        self.assertTrue(np.allclose(x, act))

    def test_one_sample(self):
        x = relu_activation(np.array([2.]))
        act = np.array([2.])
        self.assertTrue(np.allclose(x, act))


class TestReluDeriv(unittest.TestCase):

    def test_n_samples(self):
        x = relu_derivative(np.array([-1., 0., 2.]))
        act = np.array([0., 0., 1.])
        self.assertTrue(np.allclose(x, act))

    def test_one_sample(self):
        x = relu_derivative(np.array([2.]))
        act = np.array([1.])
        self.assertTrue(np.allclose(x, act))


class TestSoftplus(unittest.TestCase):

    def test_n_samples(self):
        x = softplus_activation(np.array([[-5., -1., 0., 2.]]))
        act = np.array([0.00671535, 0.31326169, 0.69314718, 2.12692801])
        self.assertTrue(np.allclose(x, act))

    def test_one_sample(self):
        x = softplus_activation(np.array([0.]))
        act = np.array([0.69314718])
        self.assertTrue(np.allclose(x, act))


class TestSoftmax(unittest.TestCase):

    def test_n_samples(self):
        x = np.array([[2.0, 1.0, 0.1],
                      [1.0, 2.0, 0.1],
                      [0.1, 1.0, 2.0],
                      [2.0, 1.0, 0.1]])
        sm = softmax_activation(x)
        act = np.array(([[0.65900114, 0.24243297, 0.09856589],
                         [0.24243297, 0.65900114, 0.09856589],
                         [0.09856589, 0.24243297, 0.65900114],
                         [0.65900114, 0.24243297, 0.09856589]]))
        self.assertTrue(np.allclose(sm, act))

    def test_one_sample(self):
        x = softmax_activation(np.array([2.0, 1.0, 0.1]))
        act = np.array([[0.65900114, 0.24243297, 0.09856589]])
        self.assertTrue(np.allclose(x, act))


if __name__ == '__main__':
    unittest.main()
