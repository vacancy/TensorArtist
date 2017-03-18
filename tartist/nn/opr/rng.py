# -*- coding:utf8 -*-
# File   : rng.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/17/17
# 
# This file is part of TensorArtist

import tensorflow as tf

from ._migrate import migrate_opr
from .helper import wrap_varnode_func
from .shape import canonize_sym_shape
import functools

__all__ = [
    'random_normal', 'random_parameterized_truncated_normal', 'random_truncated_normal',
    'random_uniform', 'random_gamma',
    'random_multinomial', 'random_shuffle'
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
random_parameterized_truncated_normal = migrate_rng_opr(tf.parameterized_truncated_normal,
                                                        name='random_parameterized_truncated_normal')
random_truncated_normal = migrate_rng_opr(tf.truncated_normal, name='random_truncated_normal')
random_uniform = migrate_rng_opr(tf.random_uniform)
random_gamma = migrate_rng_opr(tf.random_gamma)

random_multinomial = migrate_opr('random_multinomial', tf.multinomial)
random_shuffle = migrate_opr('random_shuffle', tf.random_shuffle)
