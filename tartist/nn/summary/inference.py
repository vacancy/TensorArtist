# -*- coding:utf8 -*-
# File   : train.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/27/17
# 
# This file is part of TensorArtistS

import functools
import tensorflow as tf

__all__ = ['INFERENCE_SUMMARIES', 'tensor', 'scalar', 'histogram', 'audio', 'image']
INFERENCE_SUMMARIES = 'inference_summaries'


def migrate_summary(tf_func):
    @functools.wraps(tf_func)
    def new_func(name, *args, **kwargs):
        kwargs.setdefault('collections', [INFERENCE_SUMMARIES])
        name_prefix = kwargs.pop('name_prefix', 'inference/')

        if hasattr(name, 'name'):
            name, tensor = name.name, name
            name = name_prefix + name
            return tf_func(name, tensor, *args, **kwargs)
        name = name_prefix + name
        return tf_func(name, *args, **kwargs)

    return new_func

tensor = migrate_summary(tf.summary.tensor_summary)
scalar = migrate_summary(tf.summary.scalar)
histogram = migrate_summary(tf.summary.histogram)
audio = migrate_summary(tf.summary.audio)
image = migrate_summary(tf.summary.image)
