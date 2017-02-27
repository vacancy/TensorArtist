# -*- coding:utf8 -*-
# File   : tfutils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/31/17
# 
# This file is part of TensorArtist

import tensorflow as tf

def clean_name(tensor, suffix=':0'):
    name = tensor.name
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name


def assign_variable(var, value, session, use_locking=False):
    session.run(var.assign(value, use_locking=use_locking))


def fetch_variable(var, session):
    try:
        return session.run(var)
    except tf.errors.FailedPreconditionError:
        session.run(var.initializer)
        return session.run(var)


