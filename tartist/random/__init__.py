# -*- coding:utf8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
#
# This file is part of TensorArtist.
# This file is part of NeuArtist2

from ..core.utils.nd import isndarray
from .rng import * 

import random as _random
import functools

__all__ = ['get_rng', 'reset_rng', 'reset_global_rng', 'with_rng', 'gen_seed', 'gen_rng', 'shuffle_multiarray', 'list_choice', 'list_shuffle']

__rng_meths__ = [
    # Utility functions
    'random_sample',        # Uniformly distributed floats over ``[0, 1)``.
    'bytes',                # Uniformly distributed random bytes.
    'random_integers',      # Uniformly distributed integers in a given range.
    'permutation',          # Randomly permute a sequence / generate a random sequence.
    'shuffle',              # Randomly permute a sequence in place.
    'seed',                 # Seed the random number generator.
    'choice',               # Random sample from 1-D array.

    # Compatibility functions
    'rand',                 # Uniformly distributed values.
    'randn',                # Normally distributed values.
    'randint',              # Uniformly distributed integers in a given range.

    # Univariate distributions
    'beta',                 # Beta distribution over ``[0, 1]``.
    'binomial',             # Binomial distribution.
    'chisquare',            # :math:`\\chi^2` distribution.
    'exponential',          # Exponential distribution.
    'f',                    # F (Fisher-Snedecor) distribution.
    'gamma',                # Gamma distribution.
    'geometric',            # Geometric distribution.
    'gumbel',               # Gumbel distribution.
    'hypergeometric',       # Hypergeometric distribution.
    'laplace',              # Laplace distribution.
    'logistic',             # Logistic distribution.
    'lognormal',            # Log-normal distribution.
    'logseries',            # Logarithmic series distribution.
    'negative_binomial',    # Negative binomial distribution.
    'noncentral_chisquare', # Non-central chi-square distribution.
    'noncentral_f',         # Non-central F distribution.
    'normal',               # Normal / Gaussian distribution.
    'pareto',               # Pareto distribution.
    'poisson',              # Poisson distribution.
    'power',                # Power distribution.
    'rayleigh',             # Rayleigh distribution.
    'triangular',           # Triangular distribution.
    'uniform',              # Uniform distribution.
    'vonmises',             # Von Mises circular distribution.
    'wald',                 # Wald (inverse Gaussian) distribution.
    'weibull',              # Weibull distribution.
    'zipf',                 # Zipf's distribution over ranked data.

    # Multivariate distributions
    'dirichlet',            # Multivariate generalization of Beta distribution.
    'multinomial',          # Multivariate generalization of the binomial distribution.
    'multivariate_normal',  # Multivariate generalization of the normal distribution.

    # Standard distributions
    'standard_cauchy',      # Standard Cauchy-Lorentz distribution.
    'standard_exponential', # Standard exponential distribution.
    'standard_gamma',       # Standard Gamma distribution.
    'standard_normal',      # Standard normal distribution.
    'standard_t',           # Standard Student's t-distribution.

    # Internal functions
    'get_state',            # Get tuple representing internal state of generator.
    'set_state',            # Set state of generator.
]

for m in __rng_meths__:
    def gen(meth_name):
        meth = getattr(get_rng(), meth_name) 

        @functools.wraps(meth)
        def wrapper(*args, **kwargs):
            return meth(*args, **kwargs)
        return wrapper

    globals()[m] = gen(m)


__all__.extend(__rng_meths__)

# Alias for `random_sample`.
random = random_sample
__all__.append('random')
