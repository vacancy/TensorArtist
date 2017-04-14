# -*- coding:utf8 -*-
# File   : debugpy
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/23/17
# 
# This file is part of TensorArtist

from .helper import wrap_varnode_func, wrap_simple_named_op
from ...core import get_logger
import numpy as np
import tensorflow as tf

logger = get_logger(__file__)

__all__ = ['callback_injector']


def _rms(var):
    return np.sqrt((var ** 2).mean())


def _default_log_callback(tensor, var):
    logger.info('log for {} (shape = {}): mean={}, std={}, rms={}, min={}, max={}'.format(tensor.name, var.shape,
          var.mean(), var.std(), _rms(var), var.min(), var.max()))


def _default_embed_callback(tensor, var):
    logger.info('embed for {}, access by tensor and var'.format(tensor.name))
    from IPython import embed; embed()


@wrap_simple_named_op
@wrap_varnode_func
def callback_injector(tensor, callback='log', name='callback_injector'):
    if callback == 'log':
        callback = _default_log_callback
    elif callback == 'embed':
        callback = _default_embed_callback
    assert callable(callback), 'callback must be callable'

    def py_func(var):
        callback(tensor, var)
        return np.int32(0)
    tf_op = tf.py_func(py_func, [tensor], tf.int32, stateful=False)

    with tf.control_dependencies([tf_op]):
        out = tf.identity(tensor, name='out')
    return out

