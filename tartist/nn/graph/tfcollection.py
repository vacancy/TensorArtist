# -*- coding:utf8 -*-
# File   : tfcollection.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/21/17
# 
# This file is part of TensorArtist.

import tensorflow as tf
import contextlib

__all__ = ['save_collections', 'clear_collections', 'restore_collections', 'freeze_collections']


def save_collections(keys, graph=None):
    graph = graph or tf.get_default_graph()
    return {k: graph.get_collection(k) for k in keys}


def clear_collections(keys, graph=None):
    graph = graph or tf.get_default_graph()
    for k in keys:
        del graph.get_collection_ref(k)[:]


def restore_collections(saved, graph=None):
    graph = graph or tf.get_default_graph()
    for k, v in saved.items():
        del graph.get_collection_ref(k)[:]
        graph.get_collection_ref(k).extend(v)


@contextlib.contextmanager
def freeze_collections(keys, graph=None):
    saved = save_collections(keys, graph=graph)
    yield
    restore_collections(saved, graph=graph)
