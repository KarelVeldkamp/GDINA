
import numpy as np
import torch
from itertools import combinations

def MSE(est, true):
    """
    Mean square error
    Parameters
    ----------
    est: estimated parameter values
    true: true paremters values

    Returns
    -------
    the MSE
    """
    return np.mean(np.power(est-true,2))


# function that takes in a matrix of attributes, and adds a column for each possible interactions of the other columns
# Z: BxA torch tensor of attributes, where B is the number of observations and A is the number of attributes
# returns: BxS tensor where S = 2^A-1

def expand_interactions(attributes):
    # make sure the attributes have 3 dimensions (IW samples, batch size, n_attributes)
    if len(attributes.shape) == 2:
        attributes = attributes.unsqueeze(0)

    n_iw_samples = attributes.shape[0]
    n_attributes = attributes.shape[2]
    n_effects = 2**n_attributes-1
    batch_size = attributes.shape[1]


    # Generate SxA matrix where each row represents whether each attribute is needed for each effect
    required_mask = torch.arange(1, n_effects + 1).unsqueeze(1).bitwise_and(1 << torch.arange(n_attributes)).bool()

    # repeat the matrix for each IW sample and each observation
    required_mask = required_mask.repeat((n_iw_samples, batch_size, 1, 1))  # IWxBxSxA

    # repeat the observed attribute pattern for each possible combination
    attributes = attributes.unsqueeze(2).repeat(1, 1, n_effects, 1)

    # set the observed attributes to 1 if they are not required for a pattern
    attributes[~required_mask] = 1

    # multiply over the diffent attributes, so that we get the probability of observing all necessary attributes
    effects = attributes.prod(3)


    return effects


def expand_interactions_old(attributes):
    n_attributes = attributes.shape[1]
    n_effects = 2**n_attributes-1
    batch_size = attributes.shape[0]


    # Generate SxA matrix where each row represents whether each attribute is needed for each effect
    required_mask = torch.arange(1, n_effects + 1).unsqueeze(1).bitwise_and(1 << torch.arange(n_attributes)).bool()
    # repeat the matrix for each observation
    required_mask = required_mask.repeat((batch_size, 1, 1))  # BxSxA

    # repeat the observed attribute pattern for each possible combination
    attributes = attributes.unsqueeze(1).repeat(1, n_effects, 1)

    # set the observed attributes to 1 if they are not required for a pattern
    attributes[~required_mask] = 1

    # multiply over the diffent attributes, so that we get the probability of observing all necessary attributes
    effects = attributes.prod(2)

    return effects

def Cor(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])



