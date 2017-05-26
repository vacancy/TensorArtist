# -*- coding:utf8 -*-
# File   : initializer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 5/25/17
# 
# This file is part of TensorArtist


import tensorflow as tf

__all__ = [
    'zeros_initializer', 'ones_initializer', 'constant_initializer',
    'random_uniform_initializer', 'random_normal_initializer', 'truncated_normal_initializer',
    'uniform_unit_scaling_initializer', 'variance_scaling_initializer',
    'orthogonal_initializer'
]

zeros_initializer = tf.zeros_initializer
ones_initializer = tf.ones_initializer
constant_initializer = tf.constant_initializer
random_uniform_initializer = tf.random_uniform_initializer
random_normal_initializer = tf.random_normal_initializer
truncated_normal_initializer = tf.truncated_normal_initializer
uniform_unit_scaling_initializer = tf.uniform_unit_scaling_initializer
variance_scaling_initializer = tf.variance_scaling_initializer
orthogonal_initializer = tf.orthogonal_initializer
