"""Test the Gaussian Process Classes.
"""
import numpy as np

import hiergp.gpmodel
import hiergp.kernels


def test_add_samples():
    """Test adding samples in different formats.
    """
    num_dims = 5
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (0.01, 10))
    model = hiergp.gpmodel.GPModel('model', sqe_kernel)

    # Add a single value
    model.add_samples(np.ones(num_dims), 3)
    assert np.sum(model.values) == 3
    assert np.sum(model.vectors) == 5

    # Add a single vector
    model.add_samples(np.ones(num_dims), [7.3])
    assert np.sum(model.values) == 10.3
    assert np.sum(model.vectors) == 10

    # Add multiple vectors
    np.random.seed(0)
    vectors = np.random.random((20, num_dims))
    values = np.random.random(20)
    model.add_samples(vectors, values)
    assert np.sum(model.values) == np.sum(values)+10.3


def test_one_kernel_infer():
    """Test the inference using a single SQE kernel.
    """
    np.random.seed(0)
    num_dims = 5
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (1, 10))
    model = hiergp.gpmodel.GPModel('model', sqe_kernel)
    vectors = np.random.random((20, num_dims))
    values = np.random.random(20)
    model.add_samples(vectors, values)

    # Ensure that inference of the training values gives back the same values
    assert np.allclose(model.infer(vectors)[0].ravel(), values)

    targets = np.random.random((5, num_dims))
    expected_mean = np.array([[0.40351709],
                              [1.26421069],
                              [0.74873146],
                              [0.78795475],
                              [0.95455718]])
    expected_s2 = np.array([0.00400628, 0.02435495, 0.0131116,
                            0.00145462, 0.04190979])
    mu, s2 = model.infer(targets)
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
    model.add_samples(vectors, values)

    # Ensure that inference of the training values gives back the same values
    assert np.allclose(model.infer(vectors)[0].ravel(), values)

    targets = np.random.random((5, num_dims))
    # Inference using two kernels simply adds their variances
    expected_s2 = 2*np.array([0.00400628, 0.02435495, 0.0131116,
                              0.00145462, 0.04190979])
    # The mean value should be (almost) identical to the single kernel case
    # for identical additive kernels
    expected_mean = np.array([[0.40351709],
                              [1.26421069],
                              [0.74873146],
                              [0.78795475],
                              [0.95455718]])
    mu, s2 = model.infer(targets)
    assert np.allclose(expected_s2, s2)
    assert np.allclose(expected_mean, mu)
