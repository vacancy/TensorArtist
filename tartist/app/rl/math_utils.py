# -*- coding:utf8 -*-
# File   : math_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/08/2017
# 
# This file is part of TensorArtist.

import scipy
import numpy as np


def discount_cumsum(x, gamma):
    """Compute the discounted cumulative summation of an 1-d array.
    From https://github.com/rll/rllab/blob/master/rllab/misc/special.py"""
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def discount_return(x, discount):
    """Compute the discounted return summation of an 1-d array.
    From https://github.com/rll/rllab/blob/master/rllab/misc/special.py"""
    return np.sum(x * (discount ** np.arange(len(x))))
