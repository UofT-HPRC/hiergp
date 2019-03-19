""" Gaussian Process Definitions.

The GPModel is responsible for combining data to perform inference and to fit
hyperparameters using a set of kernels.

TODO:
    Split up large infer/fit functions.
    Test transfer matrix in GPModel
"""

import logging

import numpy as np
import scipy as sp
from scipy.optimize import minimize

import hiergp.kernels

LOG = logging.getLogger(__name__)
EPS = 1e-8


def lmgrad(hypers, kernels, sampled_x, sampled_y, precomp_dx=None):
    """Compute the negative log marginal likelihood and gradient.

    For a list of kernels, compute part of the log marginal likelihood and
    the associated gradient w.r.t. each hyperparameter.

    The log marginal likelihood is:

    :math:`\\log p(y|X, \\theta) = -\\frac{1}{2} y^T K^{-1} y - \\frac{1}{2}
    \\log |K| - \\frac{n}{2} \\log 2 \\pi`

    To maximize the likelihood, this function computes:

    :math:`y^T K^{-1} y + \\log |K|`

    and the associated gradient:

    :math:`-\\text{tr}((\\alpha \\alpha^T - K^{-1})
    \\frac{\\partial K}{\\partial\\theta_j})`

    where

    :math:`\\alpha = K^{-1}y`

    The Cholesky factorization is used to compute the log determinant using
    the eigenvalues of :math:`K`.

    If kernels require different input values (e.g. after transformation),
    sampled_x should be a list of values corresponding to each kernel in the
    kernel list.

    Optionally, scale factors can be learned for prior values.
    If this current GPModel is responsible for modelling:

    :math:`y=z-\\rho_1 y_1-\\rho_2 y_2 \\dots \\rho_M y_M`

    instead of passing :math:`y` directly, the values for :math:`\\rho_i`
    can be learned along with the kernel hyperparameters.
    To do this, the sampled_y should be an array containing:

    [:math:`y_1, y_2, \\dots y_M, z`]

    The hypers must also contain :math:`\\rho_1 \\dots \\rho_M` after the
    kernel hyperparameters.

    Args:
        hypers : Flattened list of hyperparameters for all kernels and
                 the model.
        kernels : List of kernels that make up the model.
        sampled_x : List of sampled vectors for each kernel. If a single
                    array is given, it is used for all kernels.
        sampled_y : Sampled values corresponding to each vector in sampled_x.
            Optionally, this can be a list of values (see above).
        precomp_dx : Optional precomputed derivatives to be passed to each
                     kernel gradient computation.
    """
    # pylint: disable=too-many-locals, invalid-name, too-many-arguments

    # Format inputs into lists
    if precomp_dx is None:
        precomp_dx = [None]*len(kernels)

    if not isinstance(sampled_x, list):
        # This should make multiple references to the same matrix
        data_x = [sampled_x]*len(kernels)
    else:
        data_x = sampled_x
    num_samples = data_x[0].shape[0]

    # gradient vector to return
    gradient = np.empty(len(hypers))

    # Split the kernel hyperparameters into respective kernels Use the get
    # hypers command to get the length of hyperparameters For now there isn't
    # much overhead, but we may want to change this in the future.
    kernel_dims = np.cumsum([len(kernel.get_hypers()[0]) for kernel in
                             kernels])
    kernel_hypers = np.split(hypers, kernel_dims)

    # Store partial kernel matrices without the variance term
    novar_K = np.empty((len(kernels), num_samples, num_samples))
    # Full kernel matrix
    K = np.zeros((num_samples, num_samples))
    for i, kernel in enumerate(kernels):
        # Update the kernel hyperparameters
        kernel.put_hypers(kernel_hypers[i])
        novar_K[i, :, :] = kernel.eval(data_x[i], data_x[i], no_var=True)
        K += kernel.var**2 * novar_K[i, :, :]
    # Add diagonal noise
    K += np.eye(num_samples)*EPS

    # Handle scaling of prior values
    if isinstance(sampled_y, list):
        if len(kernel_hypers[-1]) != len(sampled_y)-1:
            raise ValueError(
                "Number of prior scales and Y values does not match")
        Y = sampled_y[-1] - sum(scale*sampled_y[i]
                                for i, scale in enumerate(kernel_hypers[-1]))
    else:
        Y = sampled_y

    # Compute the log marginal likelihood
    L = np.linalg.cholesky(K)
    alphaf = np.linalg.solve(L, Y).reshape(-1, 1)
    log_mlikelihood = np.dot(alphaf.T, alphaf) + 2.*sum(np.log(np.diag(L)))

    # Compute gradients
    c = sp.linalg.solve_triangular(L, np.eye(num_samples), lower=True,
                                   check_finite=False)
    Ki = np.dot(c.T, c)
    alpha = np.dot(Ki, Y).reshape(-1, 1)
    AAT = np.dot(alpha, alpha.T) - Ki

    offsets = [0] + list(kernel_dims)
    for i, kernel in enumerate(kernels):
        kernel_grad = -kernel.grad_k(data_x[i], AAT, novar_K=novar_K[i, :, :],
                                     DXX=precomp_dx[i])
        gradient[offsets[i]:offsets[i+1]] = kernel_grad

    if isinstance(sampled_y, list):
        prior_grad = np.empty(len(sampled_y)-1)
        partial_grad = np.dot(Y.T, Ki)
        for i, _ in enumerate(kernel_hypers[-1]):
            prior_grad[i] = -2*np.dot(partial_grad, sampled_y[i])
        gradient[kernel_dims[-1]:] = prior_grad

    return log_mlikelihood, gradient


class GPModel():
    """Gaussian Process Model.

    This class encapsulates the functionality to use the the covariance
    functions to perform inference and fit.

    The kernels are combined to form a composite kernel. Inputs to each kernel
    can be linearly transformed using a provided transformation matrix.



    Args:
        name (string) : name for the model
        kernels (list) : list of hiergp kernels or a single kernel
        num_priors (int, optional) : Number of mean prior samples to fit.
        y_bounds (pair, optional) : Bounds on the scale factors on priors.
        txfr_mtxs (list of arrays, optional) : List of transformations to
            apply to each kernel input. If an entry is None, the original
            data is used.
    """

    def __init__(self,
                 name,
                 kernels,
                 num_priors=0,
                 y_bounds=(0.01, 2),
                 txfr_mtxs=None):
        if not isinstance(kernels, list):
            kernels = [kernels]
        self.name = name
        self.kernels = kernels
        if not isinstance(txfr_mtxs, list):
            self.txfr_mtxs = [txfr_mtxs]*len(kernels)
        self.y_scales = np.full(num_priors, np.min(y_bounds))
        self.bounds = dict(y_scales=[y_bounds for _ in range(num_priors)])

    def infer(self, targets, sampled_x, sampled_y):
        """Compute posterior on the NxD targets using sampled data.

        This function computes:

        :math:`\\mu(Z) = K(X,Z)^T (K(X,X)+\\sigma_nI)^{-1} (Y)`

        To support a prior mean function, the Y values must already have
        the mean values subtracted.

        Args:
            targets : NxD :math:`Z`
            sampled_x : MxD :math:`X`
            sampled_y : Mx1 :math:`Y`
        """
        # Ignore 'bad' names since these correspond to equation symbols
        # pylint: disable=invalid-name

        assert sampled_x.shape[1] == targets.shape[1]

        # Perform transformation as needed
        K = np.zeros((sampled_x.shape[0], sampled_x.shape[0]))
        Ks = np.zeros((targets.shape[0], sampled_x.shape[0]))
        for i, kernel in enumerate(self.kernels):
            if self.txfr_mtxs[i] is not None:
                sampled = np.dot(sampled_x, self.txfr_mtxs[i].T)
                target = np.dot(targets, self.txfr_mtxs[i].T)
            else:
                sampled = sampled_x
                target = targets
            K += kernel.eval(sampled, sampled)
            Ks += kernel.eval(target, sampled)

        # Add noise term
        K += np.eye(K.shape[0])*EPS
        L = np.linalg.cholesky(K)

        Y = sampled_y
        LLY = np.linalg.solve(L, Y)
        Lk = np.linalg.solve(L, Ks.T)
        mu = np.dot(Lk.T, LLY)
        scales = sum(k.scale(targets) for k in self.kernels)
        s2 = scales - np.sum(Lk**2, axis=0)

        return mu, s2

    def fit(self, sampled_x, sampled_y):
        """Fit the kernel and model parameters.

        Args:
            sampled_x : NxD matrix of sampled N vectors
            sampled_y : Nx1 vector of corresponding values
        """

        if self.y_scales.shape[0] != 0:
            assert isinstance(sampled_y, list)

        # Pre-compute partial derivatives for SQE and Lin Kernels
        precomp_table = {}
        x_table = {}
        precomp_dx = []
        hypers = []
        hyper_bounds = []
        data_x = []

        for i, kernel in enumerate(self.kernels):
            hyp, hyp_bounds = kernel.get_hypers()
            hypers.append(hyp)
            hyper_bounds += hyp_bounds
            precomp_name = kernel.__class__.__name__ + \
                str(id(self.txfr_mtxs[i]))

            if precomp_name not in precomp_table:
                # TODO: Ensure this is only executed for sq or lin kernels
                if self.txfr_mtxs[i] is not None:
                    sampled = np.dot(sampled_x, self.txfr_mtxs[i].T)
                else:
                    sampled = sampled_x
                x_table[precomp_name] = sampled

                if isinstance(kernel, hiergp.kernels.SqKernel):
                    precomp = np.empty((sampled.shape[1],
                                        sampled.shape[0],
                                        sampled.shape[0]))
                    for dim in range(sampled.shape[1]):
                        dx = sampled[:, dim].reshape(-1, 1)
                        precomp[dim, :, :] = - ((dx.T - dx)**2)
                    precomp_table[precomp_name] = precomp
                elif isinstance(kernel, hiergp.kernels.LinKernel):
                    precomp = np.empty((sampled.shape[1],
                                        sampled.shape[0],
                                        sampled.shape[0]))
                    for dim in range(sampled.shape[1]):
                        dx = sampled[:, dim].reshape(-1, 1)
                        precomp[dim, :, :] = np.dot(dx, dx.T)
                    precomp_table[precomp_name] = precomp
                else:
                    precomp_table[precomp_name] = None
            data_x.append(x_table[precomp_name])
            precomp_dx.append(precomp_table[precomp_name])

        hypers = np.concatenate(hypers + [self.y_scales])
        hyper_bounds += self.bounds['y_scales']

        # Run optimization
        best_result = None
        for trial in range(3):
            # pylint: disable=unsubscriptable-object
            if trial != 0:
                hypers = best_result['x'] + np.random.randn(hypers.shape[0])
            optres = minimize(lmgrad, hypers,
                              args=(self.kernels, data_x, sampled_y,
                                    precomp_dx),
                              jac=True, tol=1e-10,
                              bounds=hyper_bounds)
            if best_result is None or optres['fun'] < best_result['fun']:
                best_result = optres
        hypers = best_result['x']

        # Store best hyperparameters back into kernels
        kernel_dims = np.cumsum([len(kernel.get_hypers()[0]) for kernel in
                                 self.kernels])
        kernel_hypers = np.split(hypers, kernel_dims)
        for i, kernel in enumerate(self.kernels):
            kernel.put_hypers(kernel_hypers[i])

        self.y_scales = hypers[kernel_dims[-1]:]
