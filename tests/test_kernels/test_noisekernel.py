"""Test Noise Kernel
"""

import pytest
import numpy as np

import hiergp.kernels

def test_variance():
    """Check the scale function.
    """
    num_points = 5

    # Create the model
    model = hiergp.kernels.NoiseKernel((0.05, 1.))
    vectors = np.ones(num_points)
    assert np.allclose(model.scale(vectors), np.ones(num_points)*model.var**2)

def test_identity():
    """Test the eval function on the same vectors.
    """
    num_points = 5
    variance = 3.

    model = hiergp.kernels.NoiseKernel((0.05, 1.))
    
    # Variance
    model.var = variance

    # Create data
    vectors = np.arange(num_points).reshape((num_points))
    assert np.allclose(model.eval(vectors, vectors), 
                       variance**2*np.eye(num_points))

