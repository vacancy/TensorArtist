# -*- coding:utf8 -*-
# File   : base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/28/17
# 
# This file is part of TensorArtist


from .. import TArtGraphKeys, opr as O, summary
from ..graph.env import get_default_env
import functools
import tensorflow as tf


__all__ = ['make_optimizer_variable', 'get_optimizer_variable,'
           'SGDOptimizer', 'MomentumOptimizer', 'AdamOptimizer', 'AdagradOptimizer', 'RMSPropOptimizer'
]


def make_optimizer_variable(name, value, prefix='', collections=TArtGraphKeys.OPTIMIZER_VARIABLES):
    name = prefix + name
    value = float(value)
    sym = O.variable(name, value, shape=0, trainable=False)
    tf.add_to_collection(collections, sym)
    summary.scalar(name, sym)
    return sym


def get_optimizer_variable(name, prefix='', env=None):
    env = env or get_default_env()
    sym = env.find_in_collection_by_name(TArtGraphKeys.OPTIMIZER_VARIABLES, prefix + name)
    return sym.taop


def _migrate_lr_based_optimizer(tf_optimizer):
    @functools.wraps(tf_optimizer)
    def opt(learning_rate, *args, **kwargs):
        if type(learning_rate) in (float, int):
            from .wrapper import OptimizerWrapper
            learning_rate = make_optimizer_variable(OptimizerWrapper.learning_rate_variable_name, learning_rate)
        return tf_optimizer(learning_rate, *args, **kwargs)
    return opt

SGDOptimizer = _migrate_lr_based_optimizer(tf.train.GradientDescentOptimizer)
MomentumOptimizer = _migrate_lr_based_optimizer(tf.train.MomentumOptimizer)
AdamOptimizer = _migrate_lr_based_optimizer(tf.train.AdamOptimizer)

AdagradOptimizer = _migrate_lr_based_optimizer(tf.train.AdagradOptimizer)
RMSPropOptimizer = _migrate_lr_based_optimizer(tf.train.RMSPropOptimizer)

