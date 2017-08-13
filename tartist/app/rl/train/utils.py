# -*- coding:utf8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/08/2017
# 
# This file is part of TensorArtist.

from tartist.nn import opr as O
from tartist.nn.graph import as_tftensor, as_varnode
from tartist.nn.tfutils import escape_name

import tensorflow as tf


def make_param_gs(env, var_list, name_scope):
    var_shapes = [as_tftensor(v).get_shape().as_list() for v in var_list]
    for vs, v in zip(var_shapes, var_list):
        assert None not in vs, 'Could not determine the shape for optimizable variable: {}.'.format(v)
    var_nr_elems = [as_tftensor(v).get_shape().num_elements() for v in var_list]
    nr_total_elems = sum(var_nr_elems)

    param_nr_elems = nr_total_elems

    with env.name_scope(name_scope):
        # Parameter getter
        flat_variables = [as_varnode(v).flatten(name='flat_{}'.format(escape_name(v))) for v in var_list]
        param_getter = as_tftensor(O.concat(flat_variables, axis=0))

        # Parameter setter
        flat_variables_tensor = O.placeholder('flat_variable_tensor', shape=(nr_total_elems, ))
        variable_assigns = []

        index = 0
        for v, vs, vn in zip(var_list, var_shapes, var_nr_elems):
            value = flat_variables_tensor[index:index+vn].reshape(vs)
            # Use tf.assign because tf.group use non-3rdparty-compatible codes.
            variable_assigns.append(tf.assign(v, value, name='assign_{}'.format(escape_name(v))))
            index += vn

        param_setter = tf.group(*variable_assigns)
        param_provider = as_tftensor(flat_variables_tensor)

    return param_nr_elems, param_getter, param_setter, param_provider
