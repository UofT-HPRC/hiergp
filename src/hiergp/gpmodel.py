""" Gaussian Process Definitions.

The GPModel is responsible for combining data to perform inference and to fit
hyperparameters.

The GPModel can be extended to handle priors.
To fit data, the GPModel must know about the priors on the samples
and to infer, it must know how to add priors.


Allow data to be passed in for inference/fit?
 - need the exact values matching the self.X and S

Two Conditions:

need to use callables?
split vectors i special case?

======================
Handling Special Cases
======================

Multi-Fidelity:
    - Prior must produce both a mean and variance to be added during inference.

Noisy Priors:
    - A prior with Gaussian noise adds noise to the 'sample'

"""

import logging

import numpy as np
import scipy as sp

import hiergp.kernels

LOG = logging.getLogger(__name__)
EPS = 1e-8


def lmgrad(hypers, kernels, sampled_X, sampled_Y, precomp_dx=None):
    """Compute the negative log marginal likelihood and gradient.

    For a list of kernels, compute part of the log marginal likelihood and
    the associated gradient w.r.t. each hyperparameter.

    The log marginal likelihood is:
    
    :math:`\\log p(y|X, \\theta) = -\\frac{1}{2} y^T K^{-1} y - \\frac{1}{2}
    \\log |K| - \\frac{n}{2} \\log 2 \\pi`

    To maximize the likelihood, this function computes:

    :math:`y^T K^{-1} y + \\log |K|`

    The Cholesky factorization is used to compute the log determinant using
    the eigenvalues of :math:`K`.



    Args:
        hypers : Flattened list of hyperparameters for all kernels and
                 the model.
        kernels : List of kernels that make up the model.
        sampled_X : List of sampled vectors for each kernel. If a single
                    array is given, it is used for all kernels.
        sampled_Y : Array of values corresponding to each sampled vector.
        precomp_dx : Optional precomputed derivatives to be passed to each
                     kernel gradient computation.
    """

    # Format inputs into lists
    if precomp_dx is None:
        precomp_dx = [None]*len(kernels)

    if not isinstance(sampled_X, list):
        # This should make multiple references to the same matrix
        data_X = [sampled_X]*len(kernels)
    else:
        data_X = sampled_X
    num_samples = data_X[0].shape[0]

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
        # TODO: Generalize to kernels without extra variance terms
        novar_K[i, :, :] = kernel.eval(data_X[i], data_X[i], no_var=True)
        K += kernel.var**2 * novar_K[i, :, :]

    # Add diagonal noise
    # FIXME: Add the noise parameter
    K += np.eye(num_samples)*EPS

    # TODO: Add exception
    # Compute the log marginal likelihood
    L = np.linalg.cholesky(K)
    alphaf = np.linalg.solve(L, sampled_Y).reshape(-1, 1)
    # This is the negative log likelihood
    log_mlikelihood = np.dot(alphaf.T, alphaf) + 2.*sum(np.log(np.diag(L)))

    # Compute gradients
    c = sp.linalg.solve_triangular(L, np.eye(num_samples), lower=True,
            check_finite=False)
    Ki = np.dot(c.T, c)
    alpha = np.dot(Ki, sampled_Y).reshape(-1, 1)
    AAT = np.dot(alpha, alpha.T) - Ki

    offsets = [0] + list(kernel_dims)
    for i, kernel in enumerate(kernels):
        kernel_grad = -kernel.grad_K(data_X[i], AAT, novar_K=novar_K[i, :, :],
                                     DXX=precomp_dx[i])
        gradient[offsets[i]:offsets[i+1]] = kernel_grad

    return log_mlikelihood, gradient


class GPModel():
    """Gaussian Process Model.

    If the parameter noise is set, a diagonal noise is tracked that can be
    added to the kernel matrix to model noisy data:

    :math:`K(X, X) + \\sigma_n I`

    Args:
        name : name for the model
        kernels : list of hiergp kernels or a single kernel
        noise(optional, float) : Track noise parameter math:`\\sigma_n` of
                                 the kernel.
    """

    def __init__(self,
                 name,
                 kernels,
                 noise=0.):
        if not isinstance(kernels, list):
            kernels = [kernels]
        assert all(k.dims == kernels[0].dims for k in kernels)
        self.name = name
        self.kernels = kernels
        self.noise = noise

    def infer(self, targets, sampled_X, sampled_Y):
        """Compute posterior on the NxD targets using sampled data.

        This function computes:

        :math:`\\mu(Z) = K(X,Z)^T (K(X,X)+\\sigma_nI)^{-1} (Y)`

        To support a prior mean function, the Y values must already have
        the mean values subtracted.

        Args:
            targets : NxD :math:`Z`
            sampled_X : MxD :math:`X`
            sampled_Y : Mx1 :math:`Y`
        """
        # Ignore 'bad' names since these correspond to equation symbols
        # pylint: disable=invalid-name

        assert sampled_X.shape[1] == targets.shape[1]

        K = sum(k.eval(sampled_X, sampled_X) for k in self.kernels)
        # Add noise term
        K += np.eye(K.shape[0])*self.noise
        K += np.eye(K.shape[0])*EPS
        L = np.linalg.cholesky(K)

        Y = sampled_Y
        LLY = np.linalg.solve(L, Y)
        Ks = sum(k.eval(targets, sampled_X) for k in self.kernels)
        Lk = np.linalg.solve(L, Ks.T)
        mu = np.dot(Lk.T, LLY)
        scales = sum(k.scale(targets) for k in self.kernels)
        s2 = scales - np.sum(Lk**2, axis=0)

        return mu, s2
