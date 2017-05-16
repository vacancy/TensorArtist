# -*- coding:utf8 -*-
# File   : distrib.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/17/17
# 
# This file is part of TensorArtist

"""
This file follows the design of tensorpack by ppwwyyxx
See https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/tfutils/distributions.py
"""

from .helper import wrap_named_class_func, lazy_O as O
import numpy as np

__all__ = [
    'DistributionBase',
    'CategoricalDistribution',
    'GaussianDistribution', 'GaussianDistributionWithUniformSample'
]


class DistributionBase(object):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @wrap_named_class_func
    def log_likelihood(self, x, theta, process_theta=False):
        assert x.ndims == 2 and x.static_shape[1] == self.sample_size, x.static_shape
        assert theta.ndims == 2 and theta.static_shape[1] == self.param_size, theta.static_shape

        if process_theta:
            theta = self._get_true_theta(theta)
        logl = self._get_log_likelihood(x, theta)

        assert logl.ndims == 1, logl.static_shape
        return O.identity(logl, name='out')

    @wrap_named_class_func
    def entropy(self, theta, process_theta=False):
        assert theta.ndims == 2 and theta.static_shape[1] == self.param_size, theta.static_shape

        if process_theta:
            theta = self._get_true_theta(theta)
        return O.identity(self._get_entropy(theta), name='out')

    @wrap_named_class_func
    def cross_entropy(self, x, theta, process_theta=False):
        log_likelihood = self.log_likelihood(x, theta, process_theta=process_theta)
        return -log_likelihood.mean(name='out')

    @wrap_named_class_func
    def sample(self, batch_size, theta, process_theta=False):
        shape = theta.static_shape
        assert len(shape) in [1, 2] and shape[-1] == self.param_size , shape
        if len(shape) == 1:
            theta = O.tile(theta.add_axis(0), [batch_size, 1])
        elif type(batch_size) is int and shape[0] is not None:
            assert shape[0] == batch_size 

        if process_theta:
            theta = self._get_true_theta(theta)
        sam = self._get_sample(batch_size, theta)
        assert sam.ndims == 2 and sam.static_shape[1] == self.sample_size, sam.static_shape
        return O.identity(sam, name='out')

    def numerical_sample(self, theta):
        shape = theta.shape
        assert len(shape) == 1 and shape[0] == self.param_size, shape
        sam = self._get_numerical_sample(theta)
        assert len(sam.shape) == 2 and sam.shape[1] == self.sample_size, sam.shape
        return sam

    @property
    def sample_size(self):
        return self._get_sample_size()

    @property
    def param_size(self):
        return self._get_param_size()

    def _get_log_likelihood(self, x, theta):
        raise NotImplementedError()

    def _get_entropy(self, theta):
        raise NotImplementedError()

    def _get_true_theta(self, theta):
        raise NotImplementedError()

    def _get_sample(self, batch_size, theta):
        raise NotImplementedError()

    def _get_numerical_sample(self, theta):
        raise NotImplementedError()

    def _get_sample_size(self):
        raise NotImplementedError()

    def _get_param_size(self):
        raise NotImplementedError()

    def __mul__(self, other):
        return MergedDistribution(self, other)


class MergedDistribution(DistributionBase):
    def __init__(self, x, y, name=None):
        name = name or '{}_X_{}'.format(x.name, y.name)
        super().__init__(name)
        self._x = x
        self._y = y

    def __split(self, var, is_param, incl_batch=True):
        length = self._x.param_size if is_param else self._x.sample_size
        if incl_batch:
            yield var[:, :length]
            yield var[:, length:]
        else:
            yield var[:length]
            yield var[length:]

    def _get_log_likelihood(self, x, theta):
        ax, bx = self.__split(x, False)
        at, bt = self.__split(theta, True)
        return self._x.log_likelihood(ax, at) + self._y.log_likelihood(bx, bt)

    def _get_entropy(self, theta):
        at, bt = self.__split(theta, True)
        return self._x.entropy(at) + self._y.entropy(bt)

    def _get_true_theta(self, theta):
        at, bt = self.__split(theta, True)
        return O.concat([self._x._get_true_theta(at), self._y._get_true_theta(bt)], axis=1)

    def _get_sample(self, batch_size, theta):
        at, bt = self.__split(theta, True)
        return O.concat([self._x.sample(batch_size, at), self._y.sample(batch_size, bt)], axis=1)

    def _get_numerical_sample(self, theta):
        at, bt = self.__split(theta, True, incl_batch=False)
        asam, bsam = self._x.numerical_sample(at), self._y.numerical_sample(bt)
        res = np.empty(shape=[asam.shape[0] * bsam.shape[0], self.sample_size], dtype='float32')
        for i in range(asam.shape[0]):
            for j in range(bsam.shape[0]):
                res[i * bsam.shape[0] + j] = np.concatenate([asam[i], bsam[j]], axis=0)
        return res

    def _get_sample_size(self):
        return self._x.sample_size + self._y.sample_size

    def _get_param_size(self):
        return self._x.param_size + self._y.param_size


class CategoricalDistribution(DistributionBase):
    _eps = 1e-8

    def __init__(self, name, nr_classes):
        super().__init__(name)
        self._nr_classes = nr_classes

    def _get_log_likelihood(self, x, theta):
        return O.reduce_sum(O.log(theta + self._eps) * x, 1)

    def _get_true_theta(self, theta):
        return O.softmax(theta)

    def _get_entropy(self, theta):
        return -O.reduce_sum(theta * O.log(theta + self._eps), 1)

    def _get_sample(self, batch_size, theta):
        ids = O.random_multinomial(O.log(theta + self._eps), num_samples=1).remove_axis(1)
        return O.one_hot(ids, self._nr_classes)

    def _get_numerical_sample(self, theta):
        return np.eye(self._nr_classes)

    def _get_sample_size(self):
        return self._nr_classes

    def _get_param_size(self):
        return self._nr_classes


class GaussianDistribution(DistributionBase):
    _eps = 1e-8
    _fixed_std_val = 1

    def __init__(self, name, size, fixed_std=True, nr_num_samples=10):
        super().__init__(name)
        self._size = size
        self._fixed_std = fixed_std
        self._nr_num_samples = nr_num_samples

    def _get_log_likelihood(self, x, theta):
        if self._fixed_std:
            mean, stddev = theta, O.ones_like(theta)
            exponent = (x - mean)
        else:
            mean, stddev = O.split(theta, 2, axis=1)
            exponent = (x - mean) / (stddev + self._eps)

        return -(0.5 * np.log(2 * np.pi) + O.log(stddev + self._eps) + 0.5 * O.sqr(exponent)).sum(axis=1)

    def _get_true_theta(self, theta):
        if self._fixed_std:
            return theta
        else:
            mean, stddev = O.split(theta, 2, axis=1)
            stddev = O.sqrt(O.exp(stddev))
            return O.concat([mean, stddev], axis=1)

    def _get_sample(self, batch_size, theta):
        if self._fixed_std:
            mean, stddev = theta, self._fixed_std_val
        else:
            mean, stddev = O.split(theta, 2, axis=1)
        e = O.random_normal(mean.shape)
        return mean + e * stddev

    def _get_entropy(self, theta):
        if self._fixed_std:
            return 0.5 * (np.log(2 * np.pi * self._fixed_std_val ** 2) + 1) * theta.shape[1].astype('float32')
        else:
            mean, stddev = O.split(theta, 2, axis=1)
            return (0.5 * np.log(2 * np.pi) + O.log(stddev + self._eps) + 0.5).sum(axis=1)

    def _get_numerical_sample(self, theta):
        assert self._size == 1
        std = theta[1] if self._fixed_std else 1
        return np.linspace(-std*3, std*3, self._nr_num_samples).reshape(-1, 1)

    def _get_param_size(self):
        return self._size if self._fixed_std else self._size * 2

    def _get_sample_size(self):
        return self._size


class GaussianDistributionWithUniformSample(GaussianDistribution):
    def __init__(self, name, size, fixed_std=True, nr_num_samples=10, min_val=-1, max_val=1, num_min_val=-2,
                 num_max_val=2):
        super().__init__(name, size, fixed_std=fixed_std, nr_num_samples=nr_num_samples)
        self._min_val, self._max_val = min_val, max_val
        self._num_min_val, self._num_max_val = num_min_val, num_max_val

    def _get_sample(self, batch_size, theta):
        return O.random_uniform([batch_size, self.sample_size], self._min_val, self._max_val)

    def _get_numerical_sample(self, theta):
        assert self._size == 1
        return np.linspace(self._num_min_val, self._num_max_val, self._nr_num_samples).reshape(-1, 1)

