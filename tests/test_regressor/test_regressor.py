"""Test the Gaussian Process Class."""
# pylint: disable=invalid-name

import numpy as np
from sklearn.linear_model import LinearRegression
import pytest

import hiergp.regressor


def test_prior():
    """Test the inference using a single SQE kernel.
    """
    np.random.seed(0)
    num_dims = 1
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (1, 10))
    reg = hiergp.regressor.GPRegressor('model', sqe_kernel,
                                       LinearRegression())


    train_x = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
    test_x = np.random.random((10,1))
    test_y = test_x*3+5
    values = train_x*3+5

    assert reg(train_x) == (0,0)

    reg.add_samples(train_x, values)

    assert np.allclose(reg(train_x)[0], values)
    assert np.allclose(reg(test_x)[0], test_y, rtol=0.1)

def test_noprior():
    """Test the inference using a single SQE kernel.
    """
    np.random.seed(0)
    num_dims = 1
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (0.01, 1))
    sqe_kernel.bounds['var'] = (0.01, 0.01)
    reg = hiergp.regressor.GPRegressor('model', sqe_kernel)

    train_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5]).reshape(-1,1)
    test_x = np.random.random((10,1))
    test_y = test_x*3+5
    values = train_x*3+5
    reg.add_samples(train_x[:3], values[:3])
    reg.add_samples(train_x[3:,:], values[3:])
    assert np.allclose(reg(train_x)[0], values, rtol=0.1)
