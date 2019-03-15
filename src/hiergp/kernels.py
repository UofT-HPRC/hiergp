""" Kernel function definitions.

A kernel class contains information used for evaluating a particular kernel
function (covariance function) between different vectors.

The kernel should manage its hyperparameters and the bounds on hyperparameters.
Compositions of kernels should be managed by a Gaussian Process Model that
contains both kernels.

In this module the Squared Exponential and Linear Kernels are defined.
"""
import logging

import numpy as np

from hiergp.fastexp import fastexp

LOG = logging.getLogger(__name__)
# TODO: Define EPS globally
EPS = 1e-8


def sqdist(vec_x, vec_y):
    """Returns a distance matrix for each pair of rows i, j in vec_x, vec_y.

    vec_x and vec_y should be 2-D matrices of shape:
    (num_points, num_dimensions)

    num_dimensions should be the same for vec_x and vec_y.
    """
    if len(vec_x.shape) == 1:
        vec_x = vec_x.reshape(1, -1)
    if len(vec_y.shape) == 1:
        vec_y = vec_y.reshape(1, -1)
    # http://stackoverflow.com/a/19094808/166749
    x_row_norms = np.einsum('ij,ij->i', vec_x, vec_x)
    y_row_norms = np.einsum('ij,ij->i', vec_y, vec_y)
    distances = np.dot(vec_x, vec_y.T)
    distances *= -2
    distances += x_row_norms.reshape(-1, 1)
    distances += y_row_norms
    distances *= -0.5
    fastexp(distances.ravel())
    return distances


class SqKernel():
    """
    This kernel implements the squared exponential function.

    The primary covariance function is the squared exponential form:

    :math:`k_s(x, x') = \\sigma_f^2 \\exp(-\\frac{1}{2} \\sum_d^D
    \\frac{(x_d-x_d')^2}{\\lambda_d^2}).`

    Args:
        dims(int) : Number of dimensions in the data (:math:`D`)
        lbound(pair) : Lower, upper bounds for lengthscales in the primary
                       kernel
    """

    def __init__(self,
                 dims,
                 lbounds):

        self.dims = dims
        self.bounds = dict(lengthscales=lbounds,
                           var=(1., 1.))

        # Initialize hyperparameters
        self.lengthscales = np.full(self.dims, lbounds[0])
        self.var = 1.

    def get_hypers(self):
        """Get flattened hyperparameters and associated bounds.

        Returns:
            Tuple : (vector of current hyperparameters,
                     list of associated bounds)
        """
        return (np.concatenate(([self.var], self.lengthscales)),
                [self.bounds['var']] +
                 [self.bounds['lengthscales'] for _ in self.lengthscales])

    def put_hypers(self, hypers):
        """Update all hyperparameters using a flattened vector.

        Note: no bounds checking of hyperparameter values is done in this
        function

        Args:
            hypers : vector of hyperparameters to insert
        """
        if len(hypers) != self.dims+1:
            raise ValueError('Incorrect number of hyperparameters to insert') 

        self.var = hypers[0]
        self.lengthscales = hypers[1:self.dims+1]

    def eval(self, vecs_1, vecs_2, no_var=False):
        """Evaluate the kernel.

        Args:
            vecs_1 (NxD array) : First matrix to compute with :math:`X`
            vecs_2 (NxD array) : Second matrix to compute with :math:`Y`
            no_var (bool) : If True, return kernel without scaling factor
                            :math:`\\sigma_f^2`
        """

        exp_distances = sqdist(vecs_1 / self.lengthscales[:self.dims],
                               vecs_2 / self.lengthscales[:self.dims])
        if no_var:
            return exp_distances
        else:
            return self.var**2 * exp_distances

    def scale(self, vecs):
        """Evaluate the scale factor for each vector in vecs.

        The scale factor is computed as :math:`k(x,x)`. Note that this
        function returns the scale factor for each vector in vec independently.
        Thus for an NxD input, a vector of size N is produced.

        The scale factor the squared exponential is just the variance factor
        :math:`\\sigma_f^2` since the squared exponential is 1. for
        identical vectors.

        Args:
            vecs (NxD array) : Vector to compute scale
        """
        return np.full(vecs.shape[0], self.var**2)


class LinKernel():
    """Linear covariance kernel.

    This kernel implements the covariance function of the form:

    :math:`k(x, x') = \\sigma_f^2 \\sum_d^D \\lambda_d x_dx'_d`

    In matrix form this is:

    :math:`K(X,Y) = X \\Lambda Y^T`
    where :math:`X` is :math:`N \\times D` and :math:`\\Lambda` is a
    diagonal :math:`D \\times D` that contains the lengthscales
    :math:`\\lambda_d`.

    Args:
        dims(int) : Number of dimensions in the data (:math:`D`)
        lbound(pair) : Lower, upper bounds for lengthscales in the primary
                       kernel
    """

    def __init__(self,
                 dims,
                 lbounds):
        self.dims = dims
        self.lbounds = lbounds

        # Initialize hyperparameters
        self.lengthscales = np.full(self.dims, lbounds[0])
        self.bounds = dict(lengthscales=lbounds,
                           var=(1., 1.))
        self.var = 1.

    def get_hypers(self):
        """Get flattened hyperparameters and associated bounds.

        Returns:
            Tuple : (vector of current hyperparameters,
                     list of associated bounds)
        """
        return (np.concatenate(([self.var], self.lengthscales)),
                [self.bounds['var']] +
                 [self.bounds['lengthscales'] for _ in self.lengthscales])

    def put_hypers(self, hypers):
        """Update all hyperparameters using a flattened vector.

        Note: no bounds checking of hyperparameter values is done in this
        function

        Args:
            hypers : vector of hyperparameters to insert
        """
        if len(hypers) != self.dims+1:
            raise ValueError('Incorrect number of hyperparameters to insert') 

        self.var = hypers[0]
        self.lengthscales = hypers[1:self.dims+1]

    def eval(self, vecs_1, vecs_2, no_var=False):
        """Evaluate the kernel.

        Args:
            vecs_1 (NxD array) : First matrix to compute with :math:`X`
            vecs_2 (NxD array) : Second matrix to compute with :math:`Y`
            no_var (bool) : If True, return kernel without scaling factor
                            :math:`\\sigma_f^2`
        """
        scaled_vec = np.dot(vecs_1, np.diag(self.lengthscales))
        lin_distances = np.dot(scaled_vec, vecs_2.T)
        if no_var:
            return lin_distances
        else:
            return self.var**2 * lin_distances

    def scale(self, vecs):
        """Evaluate the scale factor for each vector in vecs.

        The scale factor is computed as :math:`k(x,x)`. Note that this
        function returns the scale factor for each vector in vec independently.
        Thus for an NxD input, a vector of size N is produced.

        Args:
            vecs (NxD array) : Vector to compute scale
        """
        vec_lambda = np.dot(vecs, np.diag(self.lengthscales))
        return self.var**2*np.sum(vec_lambda*vecs, axis=1)
