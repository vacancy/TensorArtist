# -*- coding:utf8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
# 
# This file is part of TensorArtist

from .graph import *
from . import opr, optimizer, summary, train

import tensorflow as tf


def varnode_to_tftensor(varnode, *args, **kwargs):
    v = varnode.impl
    if isinstance(v, tf.Variable):
        return tf.convert_to_tensor(v)
    return v

tf.register_tensor_conversion_function(VarNode, varnode_to_tftensor)
