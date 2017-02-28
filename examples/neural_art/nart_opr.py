# -*- coding:utf8 -*-
# File   : nart_opr.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/27/17
# 
# This file is part of TensorArtist

from tartist.nn import opr as O
import numpy as np
import tensorflow as tf


def get_content_loss(p, x):
    c = p.shape[3]
    n = p.shape[1] * p.shape[2]
    loss = (1. / (2. * n ** 0.5 * c ** 0.5)) * ((x - p) ** 2.).sum()
    return O.as_varnode(loss)


def get_style_loss(a, x):
    c = a.shape[3]
    n = x.shape[1] * x.shape[2]
    a = a.reshape(-1, c)
    x = x.reshape(-1, c)

    ga = np.dot(a.T, a)
    gx = tf.matmul(x.dimshuffle(1, 0), x)

    a = 1. / ((4. * a.shape[0] * c ** 2.) * tf.cast(c, 'float32'))

    loss = a * tf.reduce_sum((gx - ga) ** 2)
    return O.as_varnode(loss)

