"""Test the Gaussian Process Classes.

TODO:
    Transfer kernels
    Multifidelity kernels
"""
import functools

import numpy as np
from scipy.optimize import approx_fprime

import hiergp.gpmodel
import hiergp.kernels


def test_one_kernel_infer():
    """Test the inference using a single SQE kernel.
    """
    np.random.seed(0)
    num_dims = 5
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (1, 10))
    model = hiergp.gpmodel.GPModel('model', sqe_kernel)
    vectors = np.random.random((20, num_dims))
    values = np.random.random(20)

    # Ensure that inference of the training values gives back the same values
    assert np.allclose(model.infer(vectors, vectors, values)[0].ravel(),
                       values)

    targets = np.random.random((5, num_dims))
    expected_mean = np.array([0.40351709, 1.26421069, 0.74873146, 0.78795475,
                              0.95455718])
    expected_s2 = np.array([0.00400628, 0.02435495, 0.0131116,
                            0.00145462, 0.04190979])
    mu, s2 = model.infer(targets, vectors, values)
    assert np.allclose(mu, expected_mean)
    assert np.allclose(s2, expected_s2)


def test_two_kernel_infer():
    """Test the inference of multiple SQE kernels.
    """
    np.random.seed(0)
    num_dims = 5
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (1, 10))
    sqe_kernel_2 = hiergp.kernels.SqKernel(num_dims, (1, 10))
    model = hiergp.gpmodel.GPModel('model', [sqe_kernel, sqe_kernel_2])
    vectors = np.random.random((20, num_dims))
    values = np.random.random(20)

    # Ensure that inference of the training values gives back the same values
    assert np.allclose(model.infer(vectors, vectors, values)[0].ravel(),
                       values)

    targets = np.random.random((5, num_dims))
    # Inference using two kernels simply adds their variances
    expected_s2 = 2*np.array([0.00400628, 0.02435495, 0.0131116,
                              0.00145462, 0.04190979])
    # The mean value should be (almost) identical to the single kernel case
    # for identical additive kernels
    expected_mean = np.array([0.40351709, 1.26421069, 0.74873146, 0.78795475,
                              0.95455718])
    mu, s2 = model.infer(targets, vectors, values)
    assert np.allclose(expected_s2, s2)
    assert np.allclose(expected_mean, mu)


def test_simple_lmgrad():
    np.random.seed(0)
    num_dims = 5
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (0.2, 10))
    sqe_kernel_2 = hiergp.kernels.SqKernel(num_dims, (0.1, 10))

    vectors = np.random.random((20, num_dims))
    values = np.random.random(20)

    # Test gradient with one SQE kernels
    hypers = np.array([1., 1, 2, 3, 4, 5])

    def log_marg_f(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [sqe_kernel],
                                     vectors, values)[0]

    scipy_grad = approx_fprime(
        hypers, log_marg_f, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        hypers, [sqe_kernel], vectors, values)[1]
    # Relatively weak tolerance, but for large gradient sizes it seems reasonable
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-2)

    # Test gradient with two SQE kernels
    hypers = np.array([2., 1, 2, 3, 4, 5,
                       2., 0.1, 0.2, 0.3, 0.4, 0.5])

    def log_marg_f(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [sqe_kernel, sqe_kernel_2],
                                     vectors, values)[0]
    scipy_grad = approx_fprime(
        hypers, log_marg_f, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        hypers, [sqe_kernel, sqe_kernel_2], vectors, values)[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-2)


def test_simple_lingrad():
    np.random.seed(0)
    num_dims = 5
    num_pts = 20
    lin_kernel = hiergp.kernels.LinKernel(num_dims, (0.2, 10))
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (0.2, 10))

    vectors = np.random.random((num_pts, num_dims))
    values = np.random.random(num_pts)

    # Test gradient with one SQE kernels
    hypers = np.array([0.03, 1, 2, 3, 4, 5])*1e-3

    def log_marg_f(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [lin_kernel],
                                     vectors, values)[0]
    scipy_grad = approx_fprime(
        hypers, log_marg_f, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        hypers, [lin_kernel], vectors, values)[1]
    # The linear kernel values are even larger and have worse tolerance
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)

    # Test gradient with two SQE kernels
    hypers = np.array([2., 1, 2, 3, 4, 5,
                       1., 0.001, 0.002, 0.003, 0.004, 0.005])

    def log_marg_f(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [sqe_kernel, lin_kernel],
                                     vectors, values)[0]
    scipy_grad = approx_fprime(
        hypers, log_marg_f, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        hypers, [sqe_kernel, lin_kernel], vectors, values)[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)