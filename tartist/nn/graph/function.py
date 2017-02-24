# -*- coding:utf8 -*-
# File   : function.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/31/17
# 
# This file is part of TensorArtist

import enum
import collections

import tensorflow as tf

from .node import as_tftensor
from ...core.logger import get_logger
logger = get_logger(__file__)

__all__ = ['Function']


class Function(object):
    class OutputResultType(enum.Enum):
        SINGLE = 1
        LIST = 2
        DICT = 3

    class OutputManager(object):
        def __init__(self, t, nr_outputs, names=None):
            self._type = t
            self._nr_outputs = nr_outputs
            self._output_names = names

        @classmethod
        def make(cls, outputs):
            from .node import VarNode

            if isinstance(outputs, (VarNode, tf.Tensor)):
                return cls(Function.OutputResultType.SINGLE, 1), [outputs]
            elif type(outputs) in (tuple, list):
                return cls(Function.OutputResultType.LIST, len(outputs)), list(outputs)
            elif type(outputs) in (dict, collections.OrderedDict):
                names = list(outputs.keys())
                syms = [outputs[k] for k in names]
                return cls(Function.OutputResultType.DICT, len(outputs), names), syms
            else:
                raise ValueError('unsupported output type')

        def format(self, outputs):
            outputs = outputs[:self._nr_outputs]
            if self._type == Function.OutputResultType.SINGLE:
                return outputs[0]
            elif self._type == Function.OutputResultType.LIST:
                return outputs
            else:
                return collections.OrderedDict(zip(self._output_names, outputs))

    def __init__(self, env):
        self._env = env
        self._inputs = None
        self._outputs = None
        self._output_manager = None
        self._extra_outputs = []
        self._extra_kw_modifiers = []

        self.__compiled = False

    @property
    def session(self):
        return self._env.session

    @property
    def flags(self):
        return self._env.flags

    def add_extra_outputs(self, out):
        assert not self.__compiled
        self._extra_outputs.append(out)

    def extend_extra_kw_modifiers(self, modifiers):
        self._extra_kw_modifiers.extend(modifiers)

    def compile(self, outputs, inputs=None):
        if self.__compiled:
            logger.warn('function {} already compiled'.format(self))

        self._inputs = inputs
        self._output_manager, self._outputs = Function.OutputManager.make(outputs)
        self._outputs = list(map(as_tftensor, self._outputs))
        self._outputs.extend(self._extra_outputs)
        self.__compiled = True

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            assert self._inputs is not None
            assert len(self._inputs) == len(args)
            feed_dict = dict(zip(self._inputs, args))
        else:
            feed_dict = kwargs

        for f in self._extra_kw_modifiers:
            f(feed_dict)
        feed_dict = self.__canonize_feed_dict(feed_dict)

        outputs = self.session.run(self._outputs, feed_dict=feed_dict)
        return self._output_manager.format(outputs)

    @staticmethod
    def __canonize_feed_dict(feed_dict):
        res = {}
        for k, v in feed_dict.items():
            if type(k) is str and not k.endswith(':0'):
                k += ':0'
            res[k] = v
        return res

