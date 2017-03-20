# -*- coding:utf8 -*-
# File   : grad_modifier.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/28/17
# 
# This file is part of TensorArtist

import tensorflow as tf

from ..tfutils import clean_name
from ...core.logger import get_logger
from ...core.utils.match import NameMatcher

logger = get_logger(__file__)

__all__ = [
    'GradModifierBase', 'NameBasedGradModifierBase', 'GlobalGradModifierBase',
    'LearningRateMultiplier', 'WeightDecay',
    'GradClip', 'GlobalGradClip', 'GlobalGradClipByAvgNorm'
]


class GradModifierBase(object):
    def __call__(self, grads_and_vars):
        grads_and_vars = self._do_modification(grads_and_vars)
        return grads_and_vars

    def _do_modification(self, grad_and_vars):
        raise NotImplementedError()


class NameBasedGradModifierBase(GradModifierBase):
    def __init__(self, rules=None):
        self._matcher = NameMatcher(rules)

    @property
    def matcher(self):
        return self._matcher

    def _do_modification(self, grad_and_vars):
        self._matcher.begin()

        res = []
        for g, v in grad_and_vars:
            name = clean_name(v)
            rule_v = self._matcher.match(name)
            if g is not None:
                if rule_v is not None:
                    g = self._op(g, v, rule_v)
                res.append((g, v))

        matched, unused = self._matcher.end()

        if len(matched) > 0:
            log_msgs = ['\tuse {} for {} (match: {})'.format(v, k, p) for k, p, v in matched]
            log_msgs.insert(0, 'Log grad modification for {}:'.format(self.__class__.__name__))
            logger.info('\n'.join(log_msgs))
        if len(unused) > 0:
            log_msg = 'Log grad modification for {}: unused patterns are {}'.format(self.__class__.__name__, unused)
            logger.warning(log_msg)

        return res

    def _op(self, grad, var, rule):
        raise NotImplementedError()


class GlobalGradModifierBase(GradModifierBase):
    def _do_modification(self, grad_and_vars):
        res = []
        for g, v in grad_and_vars:
            if g is not None:
                g = self._op(g, v)
                res.append((g, v))
        return res

    def _op(self, grad, var):
        raise NotImplementedError()


class LearningRateMultiplier(NameBasedGradModifierBase):
    def _op(self, grad, var, rule):
        return grad * rule


class WeightDecay(NameBasedGradModifierBase):
    def _op(self, grad, var, rule):
        return grad + var * rule


class GradClip(NameBasedGradModifierBase):
    def _op(self, grad, var, rule):
        if type(rule) in (tuple, list):
            assert len(rule) == 2, rule
            lower, upper = rule
        else:
            rule = float(rule)
            lower, upper = -rule, rule
        _ = grad
        _ = tf.maximum(_, upper)
        _ = tf.minimum(_, lower)
        return _


class GlobalGradClip(GlobalGradModifierBase):
    def __init__(self, lower, upper=None):
        if upper is None:
            lower, upper = -lower, lower
        self._lower, self._upper = lower, upper

    def _op(self, grad, var):
        _ = grad
        _ = tf.maximum(_, self._upper)
        _ = tf.minimum(_, self._lower)
        return _


class GlobalGradClipByAvgNorm(GlobalGradModifierBase):
    def __init__(self, clip_norm):
        self._clip_norm = clip_norm

    def _op(self, grad, var):
        _ = grad
        _ = tf.clip_by_average_norm(_, self._clip_norm)
        return _
