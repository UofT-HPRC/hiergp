"""Test log marginal likelihood gradient."""
# pylint: disable=invalid-name, no-member, too-many-locals

import pytest

import numpy as np
import scipy as sp
from scipy.optimize import approx_fprime

import hiergp.gpmodel
import hiergp.kernels


def test_simple_lmgrad():
    """Test the sqe kernel."""
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
    # Relatively weak tolerance, but for large gradient sizes it seems
    # reasonable
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-2)

    # Test gradient with two SQE kernels
    hypers = np.array([2., 1, 2, 3, 4, 5,
                       2., 0.1, 0.2, 0.3, 0.4, 0.5])

    def log_marg_f_2k(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [sqe_kernel, sqe_kernel_2],
                                     vectors, values)[0]
    scipy_grad = approx_fprime(
        hypers, log_marg_f_2k, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        hypers, [sqe_kernel, sqe_kernel_2], vectors, values)[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-2)


def test_simple_lingrad():
    """Test the linear kernel gradient and a combination of linear and
    sqe kernel
    """
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

    def log_marg_f_sqlin(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [sqe_kernel, lin_kernel],
                                     vectors, values)[0]
    scipy_grad = approx_fprime(
        hypers, log_marg_f_sqlin, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        hypers, [sqe_kernel, lin_kernel], vectors, values)[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)


def test_novarK_is_None():
    """Test the linear kernel gradient and a combination of linear and
    sqe kernel
    """
    np.random.seed(0)
    num_dims = 5
    num_pts = 20
    lin_kernel = hiergp.kernels.LinKernel(num_dims, (0.2, 10))
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (0.2, 10))

    vectors = np.random.random((num_pts, num_dims))
    values = np.random.random(num_pts)

    def simple_lmgrad(hypers, kernels, sampled_x, sampled_y):
        """Simple version of marginal likelihood gradient with no precomputed
        values.
        """
        num_samples = sampled_x.shape[0]

        # gradient vector to return
        gradient = np.empty(len(hypers))

        # Split the kernel hyperparameters into respective kernels Use the get
        # hypers command to get the length of hyperparameters For now there
        # isn't much overhead, but we may want to change this in the future.
        kernel_dims = np.cumsum([len(kernel.get_hypers()[0]) for kernel in
                                 kernels])
        kernel_hypers = np.split(hypers, kernel_dims)

        # Full kernel matrix
        K = np.zeros((num_samples, num_samples))
        for i, kernel in enumerate(kernels):
            kernel.put_hypers(kernel_hypers[i])
            K += kernel.eval(sampled_x, sampled_x)
        K += np.eye(num_samples)*np.sqrt(np.finfo(float).eps)

        L = np.linalg.cholesky(K)
        alphaf = np.linalg.solve(L, sampled_y).reshape(-1, 1)
        log_mlikelihood = np.dot(alphaf.T, alphaf) + 2.*sum(np.log(np.diag(L)))

        # Compute gradients
        c = sp.linalg.solve_triangular(L, np.eye(num_samples), lower=True,
                                       check_finite=False)
        Ki = np.dot(c.T, c)
        alpha = np.dot(Ki, sampled_y).reshape(-1, 1)
        AAT = np.dot(alpha, alpha.T) - Ki

        offsets = [0] + list(kernel_dims)
        for i, kernel in enumerate(kernels):
            kernel_grad = -kernel.grad_k(sampled_x, AAT)
            gradient[offsets[i]:offsets[i+1]] = kernel_grad
        return log_mlikelihood, gradient

    # Test gradient with one SQE kernels
    hypers = np.array([0.03, 1, 2, 3, 4, 5])*1e-3

    def log_marg_f(hypers):
        return simple_lmgrad(hypers, [lin_kernel],
                             vectors, values)[0]
    scipy_grad = approx_fprime(
        hypers, log_marg_f, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = simple_lmgrad(
        hypers, [lin_kernel], vectors, values)[1]
    # The linear kernel values are even larger and have worse tolerance
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)

    # Test gradient with two SQE kernels
    hypers = np.array([2., 1, 2, 3, 4, 5,
                       1., 0.001, 0.002, 0.003, 0.004, 0.005])

    def log_marg_f_sqlin(hypers):
        return simple_lmgrad(hypers, [sqe_kernel, lin_kernel],
                             vectors, values)[0]
    scipy_grad = approx_fprime(
        hypers, log_marg_f_sqlin, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = simple_lmgrad(
        hypers, [sqe_kernel, lin_kernel], vectors, values)[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)


def test_precomputed():
    """Test use of precomputed partial derivatives for SQE and Lin kernels.
    """
    np.random.seed(0)
    num_dims = 5
    num_pts = 100
    lin_kernel = hiergp.kernels.LinKernel(num_dims, (0.2, 10))
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (0.2, 10))

    vectors = np.random.random((num_pts, num_dims))
    values = np.random.random(num_pts)

    sqe_hypers = np.array([100., 0.1, 0.2, 0.3, 0.4, 0.5])
    lin_hypers = np.array([0.03, 1, 2, 3, 4, 5])*1e-3

    precompute_sqe = np.empty((vectors.shape[1],
                               vectors.shape[0], vectors.shape[0]))
    for dim in range(vectors.shape[1]):
        # Value of X at dimension 'dim' for all samples
        x = vectors[:, dim].reshape(-1, 1)
        precompute_sqe[dim, :, :] = -(x.T - x)**2

    def log_marg_f(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [sqe_kernel],
                                     vectors, values)[0]

    scipy_grad = approx_fprime(
        sqe_hypers, log_marg_f, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        sqe_hypers, [sqe_kernel], vectors, values)[1]
    gpmodel_grad_pre = hiergp.gpmodel.lmgrad(
        sqe_hypers, [sqe_kernel], vectors, values,
        precomp_dx=[precompute_sqe])[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)
    assert np.allclose(gpmodel_grad, gpmodel_grad_pre)

    precompute_lin = np.empty((vectors.shape[1],
                               vectors.shape[0], vectors.shape[0]))
    for dim in range(vectors.shape[1]):
        # Value of X at dimension 'dim' for all samples
        x = vectors[:, dim].reshape(-1, 1)
        precompute_lin[dim, :, :] = np.dot(x, x.T)

    def log_marg_f_lin(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [lin_kernel],
                                     vectors, values)[0]

    scipy_grad = approx_fprime(
        lin_hypers, log_marg_f_lin, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        lin_hypers, [lin_kernel], vectors, values)[1]
    gpmodel_grad_pre = hiergp.gpmodel.lmgrad(
        lin_hypers, [lin_kernel], vectors, values,
        precomp_dx=[precompute_lin])[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)
    assert np.allclose(gpmodel_grad, gpmodel_grad_pre)


def test_transfer_grad():
    """Test using a transfer matrix to transform vectors"""
    np.random.seed(0)
    num_dims = 5
    num_pts = 100

    # Use 3 transfer dims
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (0.2, 10))
    txfr_sqe_kernel = hiergp.kernels.SqKernel(3, (0.2, 10))

    vectors = np.random.random((num_pts, num_dims))
    values = np.random.random(num_pts)
    transfer_matrix = np.array([[1, 1, 0, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 1]])

    txfr_vecs = np.dot(vectors, transfer_matrix.T)
    hypers = np.array([5., 0.2, 0.21, 0.31, 0.4, 0.5] +
                      [3., 0.1, 0.2, 0.3])

    def log_marg_f(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [sqe_kernel, txfr_sqe_kernel],
                                     [vectors, txfr_vecs], values)[0]
    scipy_grad = approx_fprime(
        hypers, log_marg_f, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        hypers, [sqe_kernel, txfr_sqe_kernel], [vectors, txfr_vecs], values)[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)


def test_noise_grad():
    """Test Noise Kernel gradient."""
    np.random.seed(0)
    num_dims = 5
    num_pts = 20
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (0.2, 10))
    noise_kernel = hiergp.kernels.NoiseKernel((0.2, 10))

    vectors = np.random.random((num_pts, num_dims))
    values = np.random.random(num_pts)

    # Test gradient of just a noise kernel
    hypers = np.array([0.03])

    def log_marg_f(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [noise_kernel],
                                     vectors, values)[0]
    scipy_grad = approx_fprime(
        hypers, log_marg_f, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        hypers, [noise_kernel], vectors, values)[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)

    # Test gradient of a noise kernel combined with Sq. Exp.
    hypers = np.array([0.1, 1, 2, 3, 4, 5, 0.5])

    def log_marg_f_sqe(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [sqe_kernel, noise_kernel],
                                     vectors, values)[0]
    scipy_grad = approx_fprime(
        hypers, log_marg_f_sqe, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        hypers, [sqe_kernel, noise_kernel], vectors, values)[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)


def test_scale_priors_grad():
    """Test gradient of priors."""
    np.random.seed(0)
    num_dims = 5
    num_pts = 20

    vectors = np.random.random((num_pts, num_dims))
    high_fid_values = np.random.random(num_pts)
    mid_fid_values = np.random.random(num_pts)
    low_fid_values = np.random.random(num_pts)

    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (0.2, 10))
    hypers = np.array([0.03, 1, 2, 3, 4, 5, 0.1])
    values = [low_fid_values, high_fid_values]

    def log_marg_f(hypers):
        return hiergp.gpmodel.lmgrad(hypers, [sqe_kernel],
                                     vectors, values)[0]
    scipy_grad = approx_fprime(
        hypers, log_marg_f, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        hypers, [sqe_kernel], vectors, values)[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)

    # Test with two priors
    hypers = np.array([0.03, 1, 2, 3, 4, 5, 0.1, 2.1])
    values = [low_fid_values, mid_fid_values, high_fid_values]

    scipy_grad = approx_fprime(
        hypers, log_marg_f, np.sqrt(np.finfo(float).eps))
    gpmodel_grad = hiergp.gpmodel.lmgrad(
        hypers, [sqe_kernel], vectors, values)[1]
    assert np.allclose(scipy_grad, gpmodel_grad, rtol=1e-1)

    # Test valueerror assertion
    hypers = np.array([0.03, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        gpmodel_grad = hiergp.gpmodel.lmgrad(
            hypers, [sqe_kernel], vectors, values)[1]
