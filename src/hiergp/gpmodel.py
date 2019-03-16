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

import hiergp.kernels

LOG = logging.getLogger(__name__)
EPS = 1e-8

def lmgrad(hypers, kernels, sampled_X, sampled_Y):
    """Compute the log marginal likelihood and gradient.
    """
    # The minimization routine requires that the current setting be wrapped
    # into a single vector 'hypers'

    # First handle regressor hypers
    # FIXME: Move this outside the lmgrad function
    gradient = np.empty(len(hypers))
    DXX = np.empty((sampled_X.shape[1], sampled_X.shape[0], sampled_X.shape[0]))
    for d in range(sampled_X.shape[1]):
        dx = sampled_X[:, d].reshape(-1,1)
        DXX[d,:,:] = -((dx.T - dx)**2)

    # Split the kernel hyperparameters into respective kernels
    # Use the get hypers command to get the length of hyperparameters
    # For now there isn't much overhead, but we may want to change this in the
    # future.
    kernel_dims = np.cumsum([len(kernel.get_hypers()[0]) for kernel in
                             kernels])
    kernel_hypers = np.split(hypers, kernel_dims)
    
    # Compute partial kernels

    # Store partial kernel matrices without the variance term
    novar_K = np.empty((len(kernels), sampled_X.shape[0], sampled_X.shape[0]))
    # Full kernel matrix
    K = np.zeros((sampled_X.shape[0], sampled_X.shape[0]))
    for i, kernel in enumerate(kernels):
        # Update the kernel hyperparameters
        kernel.put_hypers(kernel_hypers[i])
        # TODO: Generalize to kernels without extra variance terms
        novar_K[i,:,:] = kernel.eval(sampled_X, sampled_X, no_var=True)
        K += kernel.var**2 * novar_K[i,:,:]

    # Add diagonal noise
    # FIXME: Add the noise parameter
    K += np.eye(sampled_X.shape[0])*EPS

    # TODO: Add exception
    # Compute the log marginal likelihood
    L = np.linalg.cholesky(K)
    alphaf = np.linalg.solve(L, sampled_Y).reshape(-1,1)
    # This is the negative log likelihood
    log_mlikelihood = np.dot(alphaf.T, alphaf) + 2.*sum(np.log(np.diag(L)))
    
    # Compute gradients
    Ki = np.linalg.inv(K)
    alpha = np.dot(Ki, sampled_Y).reshape(-1,1)
    AAT = np.dot(alpha, alpha.T) - Ki

    offsets = [0] + list(kernel_dims)
    for i, kernel in enumerate(kernels):
        if isinstance(kernel, hiergp.kernels.SqKernel):
            precomp_dxx = DXX
        else:
            precomp_dxx = None
        kernel_grad = -kernel.grad_K(sampled_X, AAT, novar_K=novar_K[i,:,:],
                                     DXX=precomp_dxx)
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
