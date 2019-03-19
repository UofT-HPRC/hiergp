"""Test the Gaussian Process Class."""
# pylint: disable=invalid-name

import numpy as np

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

    model.fit(vectors, values)
test_two_kernel_infer()

def test_fit():
    """Test fit function of gpmodel
    """
    np.random.seed(0)
    num_dims = 5
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (0.01, 10))
    lin_kernel = hiergp.kernels.LinKernel(num_dims, (0.01, 10))
    noise_kernel = hiergp.kernels.NoiseKernel((0.01, 0.1))
    model = hiergp.gpmodel.GPModel('model', [sqe_kernel, noise_kernel])

    # Create dataset that is correlated only to the first two dimensions
    vectors = np.random.random((20, num_dims))
    values = vectors[:, 0]*2 + vectors[:, 1]
    model.fit(vectors, values)
    assert all(model.kernels[0].lengthscales[0] <
               model.kernels[0].lengthscales[2:])
    assert all(model.kernels[0].lengthscales[1] <
               model.kernels[0].lengthscales[2:])

    # Test the Linear Kernel
    model = hiergp.gpmodel.GPModel('model', [lin_kernel, noise_kernel])
    model.fit(vectors, values)
    assert all(model.kernels[0].lengthscales[0] >
               model.kernels[0].lengthscales[2:])
    assert all(model.kernels[0].lengthscales[1] >
               model.kernels[0].lengthscales[2:])

def test_multifid_fit():
    """Test fit of multifidelity gradient.
    """
    np.random.seed(0)
    num_dims = 5
    lin_kernel = hiergp.kernels.LinKernel(num_dims, (0.01, 10))

    vectors = np.random.random((20, num_dims))
    low_values = vectors[:, 0]*2 + vectors[:, 1]
    mid_values = np.random.random(20)
    high_values = vectors[:, 0]*2 + vectors[:, 1]
    model = hiergp.gpmodel.GPModel('model', [lin_kernel], 2)
    model.bounds['y_scales'] = [(0.01, 2), (0.01, 2)]
    model.fit(vectors, [low_values, mid_values, high_values])

    assert np.allclose(model.y_scales, [1., 0.01], rtol=0.1)
    assert model.y_scales[0] > model.y_scales[1]

def test_mu_fit():
    """One way to fit a generic mean function is to use a unit value for 
    other y values.
    """
    num_dims = 1
    vectors = np.random.random((20, num_dims))
    values = np.random.randn(20)*0.01 + 8.
    sqe_kernel = hiergp.kernels.SqKernel(num_dims, (0.01, 10))
    sqe_kernel.bounds['var'] = (0.01, 0.01)
    model = hiergp.gpmodel.GPModel('model', sqe_kernel, num_priors=1)
    model.bounds['y_scales'] = [(0.1, 10)]
    model.fit(vectors, [np.ones(20), values])

    assert np.isclose(model.y_scales, 8., rtol=0.1)

def test_transfer_fit():
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
    model = hiergp.gpmodel.GPModel('model', [sqe_kernel, txfr_sqe_kernel],
                                   txfr_mtxs=[None, transfer_matrix])
    assert np.allclose(model.infer(vectors, vectors, values)[0],values)
    model.fit(vectors, values)
    assert np.allclose(model.infer(vectors, vectors, values)[0],values, rtol=0.1)
