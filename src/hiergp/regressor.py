"""Example regressor types.

A regressor keeps track of the training and handles model updates.
This allows the regressor to update sampled values to be fed into a GPModel
based on updates to mean function parameters.
"""
import logging

import numpy as np

import hiergp.gpmodel

LOG = logging.getLogger(__name__)


class InferResult():
    def __init__(self, mu, s2, extras={}):
        self.mu = mu
        self.s2 = s2
        self.extras = extras


class GPRegressor():
    """Gaussian Process Regressor with prior.

    Args:
        name : Name to track the model
        kernels : List of kernels to be part of the GPModel. The first kernel
            must
        prior (regressor, optional) : Mean function that acts as a prior
            belief.  The prior should be an object that implements fit() and
            predict() functions such as those in the Scikit learn library.
    """

    def __init__(self,
                 name,
                 kernels,
                 prior=None):
        self.name = name
        self.gpmodel = hiergp.gpmodel.GPModel(name, kernels, num_priors=1)
        self.reg = prior

        # Initialize training data
        self.sampled_x = None
        self.sampled_y = np.empty((0, 1))

        self._stale_factors = True
        self._factors = None

    def add_samples(self, sampled_x, sampled_y):
        """Add samples to the model.

        Args:
            sampled_x (numpy array) : NxD array of vectors to add
            sampled_y (numpy array) : N values associated with vector values
        """
        if self.sampled_x is None:
            self.sampled_x = sampled_x
        else:
            self.sampled_x = np.vstack((self.sampled_x, sampled_x))
        self.sampled_y = np.vstack((self.sampled_y,
                                    np.array(sampled_y).reshape((-1, 1))))

        # Fit prior
        if self.reg is not None:
            self.reg.fit(self.sampled_x, self.sampled_y)
            prior_y = self.reg.predict(self.sampled_x)
        else:
            prior_y = 0.

        # Fit GPModel
        # gp_y = self.sampled_y - prior_y
        # alpha = 10 + self.sampled_y.shape[0]/2
        # beta = (np.mean(self.sampled_y)**2 +
        #         0.5*np.sum((gp_y-np.mean(gp_y))**2))
        # if beta <= 0:
        #     beta = 0.

        var_est = np.abs(np.mean(self.sampled_y))

        for kernel in self.gpmodel.kernels:
            if kernel.__class__.__name__ == "SqKernel":
                kernel.bounds['var'] = (var_est, None)
            else:
                kernel.bounds['var'] = (0.1*var_est, None)
        self.gpmodel.fit(self.sampled_x, [self.sampled_y, prior_y])
        for kernel in self.gpmodel.kernels:
            LOG.info(f"{self.name} {kernel.__class__.__name__} "
                     f"{kernel.var} {var_est} {np.mean(self.sampled_y)}")

    def infer(self, vectors):
        """Compute posterior using prior and Gaussian process.

        Args:
            vectors (numpy array) : NxD array of vectors to run inference on

        Returns:
            A tuple (mean, variance) where each is a numpy array corresponding
            to the distribution parameters for each of the N inputs.
        """
        if self.sampled_x is None:
            return InferResult(np.zeros(1), np.zeros(1))
            # raise RuntimeError("This model requires data before use")

        # Compute the prior value
        if self.reg is None:
            prior_y = 0.
            prior_sampled_y = 0.
        else:
            prior_y = self.reg.predict(vectors)
            prior_sampled_y = self.reg.predict(self.sampled_x)
        prior_y = prior_y * self.gpmodel.y_scales[0]
        prior_sampled_y = prior_sampled_y * self.gpmodel.y_scales[0]
        means, variances = self.gpmodel.infer(vectors,
                                              self.sampled_x,
                                              self.sampled_y-prior_sampled_y)
        means += prior_y.ravel()
        return InferResult(means, variances, {'prior_y': prior_y})

    @property
    def factors(self):
        if self._stale_factors:
            self._stale_factors = False
        else:
            return self._factors

    def __call__(self, vectors):
        """A call of this type runs inference."""
        return self.infer(vectors)
