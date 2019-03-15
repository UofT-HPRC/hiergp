"""Test Squared Exponential Kernel
"""

import numpy as np

import hiergp.kernels

def test_variance():
    """Check the scale function.

    The scale function is the squared L2 norm of the vector times the model
    variance.
    """
    num_points = 5
    num_dims = 3

    # Create the model
    model = hiergp.kernels.LinKernel(num_dims, (0.05, 1.))
    model.lengthscales = np.ones(num_dims)
    model.var = 1.

    vectors = np.ones((num_points, num_dims))
    assert np.allclose(model.scale(vectors),
                       num_dims*np.ones(num_points)*model.var**2)

def test_small_eval():
    """Run the evaluation function against different sizes.
    """
    num_points = 1
    num_dims = 5
    variance = 3.

    model = hiergp.kernels.LinKernel(num_dims, (0.05, 1.))

    # Test a single vector shape (1,5)
    # Lengthscales: small values force the kernel to be roughly identity
    model.lengthscales = np.full(num_dims, 0.001)
    # Variance
    model.var = variance

    # Create data
    vectors = np.arange(num_points*num_dims).reshape((num_points, num_dims))
    sv = np.dot(vectors, np.diag(model.lengthscales))
    assert np.allclose(model.scale(vectors).ravel(),
                       np.diag(model.eval(vectors, vectors)).ravel())
    assert np.allclose(model.eval(vectors, vectors).ravel(), 0.27)

    # Ensure eval works with both (D,) and (N,D) shapes
    model.lengthscales = np.full(num_dims, 10.)
    vectors_1 = np.array([0, 1, 0, 0, 1])
    vectors_2 = np.array([[1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1]])
    expected_eval = np.array([90., 180.])
    result_12 =  model.eval(vectors_1, vectors_2)
    result_21 =  model.eval(vectors_2, vectors_1)
    assert np.allclose(result_12.ravel(), expected_eval)
    assert np.allclose(result_21.ravel(), expected_eval)

    # Test with larger vectors
    vectors_1 = np.array([[0, 0.11, 0, 0.3, 1],
                          [3, 1, 2, 0.22, 1]])
    vectors_2 = np.array([[1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1]])
    expected_eval = np.array([[ 90., 126.9],
                              [540., 649.8]])
    result_12 =  model.eval(vectors_1, vectors_2)
    result_21 =  model.eval(vectors_2, vectors_1)
    assert np.allclose(result_12, expected_eval)
    assert np.allclose(result_21.T, expected_eval)

def test_hypers():
    """Check the get_hypers and put_hypers functions
    """
    num_points = 1
    num_dims = 5
    variance = 3.

    model = hiergp.kernels.SqKernel(num_dims, (0.05, 1.))

    hypers, bounds = model.get_hypers()
    assert len(hypers) == len(bounds)
    assert all([hyper >= bounds[i][0] or 
                hyper <= bounds[i][1] for i, hyper in 
                enumerate(hypers)])

    # Update hyperparameters

    hypers = np.array([10., 0.8, 0.9, 0.3, 0.25, 0.1])
    model.put_hypers(hypers)

    new_hypers, _ = model.get_hypers()
    assert all(hypers == new_hypers)
