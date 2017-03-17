# -*- coding:utf8 -*-
# File   : tfutils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/31/17
# 
# This file is part of TensorArtist

import re
import tensorflow as tf


class TArtGraphKeys:
    TART_OPERATORS = 'tart_operators'
    INFERENCE_SUMMARIES = 'inference_summaries'
    OPTIMIZER_VARIABLES = 'optimizer_variables'


def clean_name(tensor, suffix=':0'):
    name = tensor.name
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name


def clean_summary_name(name):
    return re.sub('_\d+$', '', name)


def assign_variable(var, value, session=None, use_locking=False):
    from .graph.env import get_default_env
    session = session or get_default_env().session
    session.run(var.assign(value, use_locking=use_locking))


def fetch_variable(var, session=None):
    from .graph.env import get_default_env
    session = session or get_default_env().session
    try:
        return session.run(var)
    except tf.errors.FailedPreconditionError:
        session.run(var.initializer)
        return session.run(var)

