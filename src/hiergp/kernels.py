""" Kernel function definitions.

A kernel class contains information used for evaluating a particular kernel
function (covariance function) between different vectors.

The kernel should manage its hyperparameters and the bounds on hyperparameters.
Compositions of kernels should be managed by a Gaussian Process Model that
contains both kernels.

In this module the Squared Exponential and Linear Kernels are defined.

TODO:
    - Reduce redundant code between kernels
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
        dims(int) : Number of lengthscale parameters :math:`D`.
        lbound(pair) : Lower, upper bounds for lengthscales in the primary
                       kernel.
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

    def grad_k(self, sampled_x, A, novar_K=None, DXX=None):
        """Compute the trace of A * Jacobian for each parameter.

        This function is specialized to work with maximizing the log
        marginal likelihood. To that end, it computes:

        :math:`tr(A \\frac{\\partial K}{\\partial \\theta_j}`

        Args:
            sampled_x (NxD) : Sampled vectors
            A (NxN) : A matrix to multiply with
            novar_K (optional) : If the kernel matrix has been precomputed,
                without the variance multiplied to it, use this instead
                of recomputing it
            DXX (optional) : If the partial derivative term has been
                precomputed, use it instead of recomputing
        """
        gradient = np.empty(self.dims+1)

        # Compute the kernel matrix if it is not given
        if novar_K is None:
            novar_K = self.eval(sampled_x, sampled_x, True)

        # Compute the squared distance if not given
        if DXX is None:
            DXX = np.empty((sampled_x.shape[1],
                            sampled_x.shape[0], sampled_x.shape[0]))
            for d in range(sampled_x.shape[1]):
                dx = sampled_x[:, d].reshape(-1, 1)
                DXX[d, :, :] = -((dx.T - dx)**2)

        # The standard deviation scales the kernel matrix
        gradient[0] = np.einsum('ij,ij->', A, 2.*self.var*novar_K)

        # Derivatves of lengthscales
        AsK = A*self.var**2*novar_K
        for d in range(sampled_x.shape[1]):
            gradient[d+1] = -(np.einsum('ij,ij->', AsK, DXX[d, :, :]) /
                              self.lengthscales[d]**3)
        return gradient

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

    def grad_k(self, sampled_x, A, novar_K=None, DXX=None):
        """Compute the trace of A * Jacobian for each parameter.

        This function is specialized to work with maximizing the log
        marginal likelihood. To that end, it computes:

        :math:`tr(A \\frac{\\partial K}{\\partial \\theta_j}`

        Args:
            sampled_x (NxD) : Sampled vectors
            A (NxN) : A matrix to multiply with
            novar_K (optional) : If the kernel matrix has been precomputed,
                without the variance multiplied to it, use this instead
                of recomputing it
            DXX (optional) : If the partial derivative term has been
                precomputed, use it instead of recomputing
        """
        gradient = np.empty(self.dims+1)
        # Compute the kernel matrix if it is not given
        if novar_K is None:
            novar_K = self.eval(sampled_x, sampled_x, True)

        # Compute the squared distance if not given
        if DXX is None:
            DXX = np.empty((sampled_x.shape[1],
                            sampled_x.shape[0],
                            sampled_x.shape[0]))
            for d in range(sampled_x.shape[1]):
                dx = sampled_x[:, d].reshape(-1, 1)
                DXX[d, :, :] = np.dot(dx, dx.T)

        # The standard deviation scales the kernel matrix
        gradient[0] = np.einsum('ij,ij->', A, 2.*self.var*novar_K)

        # Derivatves of lengthscales
        AsK = A*self.var**2
        for d in range(sampled_x.shape[1]):
            gradient[d+1] = np.einsum('ij,ij->', AsK, DXX[d, :, :])
        return gradient

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


class NoiseKernel():
    """Noise kernel.

    This function is equivalent to adding i.i.d Gaussian noise with zero mean
    to the values.

    The kernel function is an identity:
    :math:`k(x,x') = \\sigma_n^2 1_{x=x'}`
    in other words the function is :math:`\\sigma_n^2` only when :math:`x=x'`.

    In matrix form the resulting kernel matrix is:

    :math:`\\sigma_n^2 I`

    This kernel is only evaluated when computiong pairwise covariance between
    points. The kernel value is zero when computing the value between different
    vectors. This allows the kernel function to be used when computing the
    mean:

    :math:`K(X,Z) (K(X,X) + \\sigma_n^2 I)^{-1} (Y)`

    Also set the scale() output to zero to support

    :math:`K(X,X) - K(X,Z)(K(X,X) + \\sigma_n^2I)^{-1}K(Z,X)`

    Args:
        var_bounds (pair) : (Lower, Upper) bounds on the noise term
    """

    def __init__(self, var_bounds):
        self.bounds = dict(var=var_bounds)
        self.var = self.bounds['var'][0]

    def get_hypers(self):
        """Get flattened hyperparameters and associated bounds.

        Returns:
            Tuple : (list containing noise parameter,
                     list of associated bound)
        """
        # Wrap the variance in a list since it is expected in lmgrad
        return ([self.var], [self.bounds['var']])

    def put_hypers(self, hypers):
        """Update variance given a numpy array.

        Args:
            hypers : (1, ) numpy array
        """
        self.var = hypers[0]

    def grad_k(self, sampled_x, A, novar_K=None, DXX=None):
        """Compute the trace of A * Jacobian for each parameter.

        This function is specialized to work with maximizing the log
        marginal likelihood. To that end, it computes:

        :math:`tr(A \\frac{\\partial K}{\\partial \\theta_j}`

        Args:
            sampled_x (NxD) : Sampled vectors
            A (NxN) : A matrix to multiply with
            novar_K (optional) : If the kernel matrix has been precomputed,
                without the variance multiplied to it, use this instead
                of recomputing it
            DXX (unused)
        """
        gradient = np.array([2.*self.var*np.trace(A)])
        return gradient

    def eval(self, vecs_1, vecs_2, no_var=False):
        """Evaluate the kernel.

        Args:
            vecs_1 (NxD array) : First matrix to compute with :math:`X`
            vecs_2 (NxD array) : Second matrix to compute with :math:`Y`
            no_var (bool) : If True, return kernel without scaling factor
                            :math:`\\sigma_f^2`
        """
        if vecs_1 is not vecs_2:
            return 0.
        if no_var:
            return np.eye(vecs_1.shape[0])
        else:
            return self.var**2*np.eye(vecs_1.shape[0])

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
        return np.zeros(vecs.shape[0])  # np.full(vecs.shape[0], self.var**2)
