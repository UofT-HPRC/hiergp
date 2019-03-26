"""Example regressor types.

A regressor keeps track of the training and handles model updates.
This allows the regressor to update sampled values to be fed into a GPModel
based on updates to mean function parameters.
"""

import numpy as np

import hiergp.gpmodel


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
        self.gpmodel = hiergp.gpmodel.GPModel(name, kernels)
        self.reg = prior

        # Initialize training data
        self.sampled_x = None
        self.sampled_y = np.empty((0, 1))

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
        self.gpmodel.fit(self.sampled_x, self.sampled_y-prior_y)

    def infer(self, vectors):
        """Compute posterior using prior and Gaussian process.

        Args:
            vectors (numpy array) : NxD array of vectors to run inference on

        Returns:
            A tuple (mean, variance) where each is a numpy array corresponding
            to the distribution parameters for each of the N inputs.
        """
        if self.sampled_x is None:
            return np.zeros(1), np.zeros(1)
            # raise RuntimeError("This model requires data before use")

        # Compute the prior value
        if self.reg is None:
            prior_y = 0.
            prior_sampled_y = 0.
        else:
            prior_y = self.reg.predict(vectors)
            prior_sampled_y = self.reg.predict(self.sampled_x)

        means, variances = self.gpmodel.infer(vectors,
                                              self.sampled_x,
                                              self.sampled_y-prior_sampled_y)
        means += prior_y

        return means, variances

    def __call__(self, vectors):
        """A call of this type runs inference."""
        return self.infer(vectors)
