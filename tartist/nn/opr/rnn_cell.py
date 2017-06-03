# -*- coding:utf8 -*-
# File   : rnn_cell.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 5/25/17
# 
# This file is part of TensorArtist.

from tartist.nn.graph.node import as_tftensor
from tartist.nn.opr.helper import wrap_varnode_func

import functools
import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn


def migrate_cell(cell_cls):
    class MigratedRNNCell(cell_cls):
        @functools.wraps(cell_cls.__call__)
        @wrap_varnode_func
        def __call__(self, inputs, state, *args, **kwargs):
            inputs = as_tftensor(inputs)
            print(inputs, state)
            # state = as_tftensor(state)
            res = super().__call__(self, inputs, state, *args, **kwargs)
            return res

    functools.update_wrapper(MigratedRNNCell, cell_cls, updated=[])

    return cell_cls


BasicRNNCell = migrate_cell(tf_rnn.BasicRNNCell)
BasicLSTMCell = migrate_cell(tf_rnn.BasicLSTMCell)
RNNCell = migrate_cell(tf_rnn.RNNCell)
LSTMCell = migrate_cell(tf_rnn.LSTMCell)
GRUCell = migrate_cell(tf_rnn.GRUCell)

__all__ = ['BasicRNNCell', 'BasicLSTMCell', 'RNNCell', 'LSTMCell', 'GRUCell']
