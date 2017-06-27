# -*- coding:utf8 -*-
# File   : rng.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/17/17
# 
# This file is part of TensorArtist.

import tensorflow as tf
from tensorflow.python.ops import random_ops

from .migrate import migrate_opr
from .helper import as_varnode, wrap_varnode_func, wrap_named_op, lazy_O as O
from .shape import canonize_sym_shape
import functools

__all__ = [
    'random_normal', 'random_parameterized_truncated_normal', 'random_truncated_normal',
    'random_uniform', 'random_gamma',
    'random_multinomial', 'random_shuffle',
    'random_bernoulli'
]


def migrate_rng_opr(tf_func, name=None):
    @functools.wraps(tf_func)
    @wrap_varnode_func
    def new_func(shape, *args, **kwargs):
        shape = canonize_sym_shape(shape)
        return tf_func(shape, *args, **kwargs)

    if name is not None:
        new_func.__name__ = name

    return new_func

random_normal = migrate_rng_opr(tf.random_normal)
random_parameterized_truncated_normal = migrate_rng_opr(random_ops.parameterized_truncated_normal,
                                                        name='random_parameterized_truncated_normal')
random_truncated_normal = migrate_rng_opr(tf.truncated_normal, name='random_truncated_normal')
random_uniform = migrate_rng_opr(tf.random_uniform)
random_gamma = migrate_rng_opr(tf.random_gamma)

random_multinomial = migrate_opr('random_multinomial', tf.multinomial)
random_shuffle = migrate_opr('random_shuffle', tf.random_shuffle)


@wrap_named_op
def random_bernoulli(logits, num_samples, seed=None, squeeze=False, name='random_bernoulli'):
    """Draw sample from a bernoulli distribution.
    Input logits is expect to be the logit(prob).
    """
    logits = as_varnode(logits)

    shape = logits.shape
    zeros_parts = logits.flatten().add_axis(1)
    ones_parts = O.zeros_like(zeros_parts)
    logits = O.concat([zeros_parts, ones_parts], axis=1)
    samples = random_multinomial(logits, num_samples, seed=seed)

    if squeeze and num_samples == 1:
        pass
    else:
        shape = O.concat([shape, [num_samples]], axis=0)

    return samples.reshape(shape)
