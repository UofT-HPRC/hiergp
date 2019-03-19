# Hierarchical Gaussian Process Library

This is a simple Gaussian process library that supports different compositions
of kernels.

It is designed to support inference and fit of hyperparameters for

* Composition of Squared Exponential and Linear kernels
* Mixture of kernels with linear input transformations on the data
* Regression with a linear combination of mean functions
  * Scale factors can be learned along with all other kernel hyperparameters
  * A simple constant prior can be learned by using a constant value (e.g. 1)
    as a mean function
* Zero mean constant variance noise on sampled data through the composition of
  a noise kernel
