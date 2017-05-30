# -*- coding:utf8 -*-
# File   : initializer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 5/25/17
# 
# This file is part of TensorArtist


from tensorflow.python.ops import init_ops

__all__ = [
    'zeros_initializer', 'ones_initializer', 'constant_initializer',
    'random_uniform_initializer', 'random_normal_initializer', 'truncated_normal_initializer',
    'uniform_unit_scaling_initializer', 'variance_scaling_initializer',
    'orthogonal_initializer'
]

zeros_initializer = init_ops.zeros_initializer
ones_initializer = init_ops.ones_initializer
constant_initializer = init_ops.constant_initializer
random_uniform_initializer = init_ops.random_uniform_initializer
random_normal_initializer = init_ops.random_normal_initializer
truncated_normal_initializer = init_ops.truncated_normal_initializer
uniform_unit_scaling_initializer = init_ops.uniform_unit_scaling_initializer
variance_scaling_initializer = init_ops.variance_scaling_initializer
orthogonal_initializer = init_ops.orthogonal_initializer
