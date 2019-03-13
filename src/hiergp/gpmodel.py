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

import numpy as np

EPS = 1e-8


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

        self.vectors = np.empty((0, self.kernels[0].dims))
        self.values = np.empty((0, 1))

    def add_samples(self, vectors, values):
        """Add samples to the model.

        Args:
            vectors : NxD array of vectors
            values : N values associated with vector values
        """
        self.vectors = np.vstack((self.vectors, vectors))
        self.values = np.vstack((self.values,
                                 np.array(values).reshape((-1, 1))))

    def infer(self, targets):
        """Compute posterior on the NxD vectors.

        This function computes:

        :math:`\\mu(Z) = \\mu_0(Z) + K(X,Z)^T (K(X,X)+\\sigma_nI)^{-1}
        (Y-\\mu_0(X))`

        Args:
            targets : NxD
        """
        # Ignore 'bad' names since these correspond to equation symbols
        # pylint: disable=invalid-name

        K = sum(k.eval(self.vectors, self.vectors) for k in self.kernels)
        # Add noise term
        K += np.eye(K.shape[0])*self.noise
        K += np.eye(K.shape[0])*EPS
        L = np.linalg.cholesky(K)

        Y = self.values
        LLY = np.linalg.solve(L, Y)
        Ks = sum(k.eval(targets, self.vectors) for k in self.kernels)
        Lk = np.linalg.solve(L, Ks.T)
        mu = np.dot(Lk.T, LLY)
        scales = sum(k.scale(targets) for k in self.kernels)
        s2 = scales - np.sum(Lk**2, axis=0)

        return mu, s2
