"""Test Squared Exponential Kernel
"""

import numpy as np

import hiergp.kernels

def test_variance():
    """Check the scale function.

    The scale function should just be the model variance squared.
    """
    num_points = 5
    num_dims = 3

    # Create the model
    model = hiergp.kernels.SqKernel(num_dims, (0.05, 1.))
    vectors = np.ones((num_points, num_dims))
    assert np.allclose(model.scale(vectors), np.ones(num_points)*model.var**2)

def test_identity():
    """Evaluate the kernel with small lengthscales.

    We expect any difference in any dimension to push the kernel function
    to approximately zero. K(X,X) should then be roughly the identity matrix
    multiplied by the variance factor.
    """
    num_points = 5
    num_dims = 3
    variance = 3.

    model = hiergp.kernels.SqKernel(num_dims, (0.05, 1.))
    
    # Lengthscales: small values force the kernel to be roughly identity
    model.lengthscales = np.full(num_dims, 0.001)
    # Variance
    model.var = variance

    # Create data
    vectors = np.arange(num_points*num_dims).reshape((num_points, num_dims))
    assert np.allclose(model.eval(vectors, vectors), 
                       variance**2*np.eye(num_points))

def test_small_eval():
    """Run the evaluation function against different sizes.
    """
    num_points = 1
    num_dims = 5
    variance = 3.

    model = hiergp.kernels.SqKernel(num_dims, (0.05, 1.))

    # Test a single vector shape (1,5)
    # Lengthscales: small values force the kernel to be roughly identity
    model.lengthscales = np.full(num_dims, 0.001)
    # Variance
    model.var = variance

    # Create data
    vectors = np.arange(num_points*num_dims).reshape((num_points, num_dims))
    assert np.allclose(model.eval(vectors, vectors), 
                       variance**2*np.eye(num_points))

    # Ensure eval works with both (D,) and (N,D) shapes
    model.lengthscales = np.full(num_dims, 10.)
    vectors_1 = np.array([0, 0, 0, 0, 1])
    vectors_2 = np.array([[1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1]])
    expected_eval = np.array([8.9104485, 8.82178806])
    result_12 =  model.eval(vectors_1, vectors_2)
    result_21 =  model.eval(vectors_2, vectors_1)
    assert np.allclose(result_12.ravel(), expected_eval)
    assert np.allclose(result_21.ravel(), expected_eval)

    # Test with larger vectors
    vectors_1 = np.array([[0, 0, 0, 0, 1],
                          [3, 1, 2, 0, 1]])
    vectors_2 = np.array([[1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1]])
    expected_eval = np.array([[8.9104485, 8.82178806],
                              [8.7340098, 8.7340098 ]])
    result_12 =  model.eval(vectors_1, vectors_2)
    result_21 =  model.eval(vectors_2, vectors_1)
    assert np.allclose(result_12, expected_eval)
    assert np.allclose(result_21.T, expected_eval)

