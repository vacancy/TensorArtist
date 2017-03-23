# -*- coding:utf8 -*-
# File   : linalg.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/22/17
# 
# This file is part of TensorArtist


from ..graph.node import as_tftensor
from .helper import wrap_varnode_func, wrap_simple_named_op
import tensorflow as tf

__all__ = ['batch_matmul']


@wrap_simple_named_op
@wrap_varnode_func
def batch_matmul(a, b, name='batch_matmul'):
    with tf.name_scope(name):
        return tf.einsum('aij,ajk->aik', as_tftensor(a), as_tftensor(b))

