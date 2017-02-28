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
    C = p.shape[1]
    N = p.shape[2] * p.shape[3]
    loss = (1. / (2 * N ** 0.5 * C ** 0.5)) * ((x - p) ** 2).sum()
    return O.as_varnode(loss)


def get_style_loss(a, x):
    c = a.shape[1]
    a = a.reshape(c, -1)
    x = x.reshape(c, -1)

    ga = np.dot(a, a.T)
    gx = tf.matmul(x, x.dimshuffle(1, 0))

    a = 1. / ((4 * a.shape[1] * c ** 2) * x.partial_shape[1])

    loss = a * ((gx - ga) ** 2).sum()
    return O.as_varnode(loss)
