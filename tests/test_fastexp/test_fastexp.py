""" Check values and input handling of fastexp
"""

import hiergp.fastexp
import numpy as np
import pytest

def test_integer():
    """Fastexp only accepts numpy arrays of doubles."""
    A = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        hiergp.fastexp.fastexp(A)

def test_random():
    """Compare output of fastexp with numpy."""
    np.random.seed(0)
    values = np.random.randn(1000)
    np_exp_values = np.exp(values)
    hiergp.fastexp.fastexp(values)
    assert np.allclose(np_exp_values, values)
