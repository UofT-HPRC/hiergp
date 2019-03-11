"""Test utility functions
"""

import numpy as np
from scipy.spatial.distance import cdist

import hiergp.kernels


def test_sqdist():
    """Compare the output of sqdist to the SciPy distance function.

    This is just a sanity check for the distance and exp functions.
    """
    np.random.seed(0)
    dimensions = 50
    num_vectors_a = 100
    num_vectors_b = 3000

    vec_a = np.random.random((num_vectors_a, dimensions))
    vec_b = np.random.random((num_vectors_b, dimensions))

    sp_distances = np.exp(-0.5*cdist(vec_a, vec_b, 'sqeuclidean'))
    hgp_distances = hiergp.kernels.sqdist(vec_a, vec_b)

    assert np.allclose(sp_distances, hgp_distances)
