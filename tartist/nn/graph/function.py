# -*- coding:utf8 -*-
# File   : function.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/31/17
#
# This file is part of TensorArtist

from .node import __valid_tensor_types__, as_tftensor, as_varnode
from ...core.logger import get_logger
from ...core.utils.meta import merge_iterable, notnone_property
logger = get_logger(__file__)

import enum
import collections

import numpy as np
import tensorflow as tf

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

            if isinstance(outputs, __valid_tensor_types__):
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

        def reduce_format(self, all_vars, all_outputs, reduce_ratios=None):
            assert len(all_vars) == self._nr_outputs
            all_outputs = [o[:self._nr_outputs] for o in all_outputs]
            ret = []
            for o, *vals in zip(all_vars, *all_outputs):
                ret.append(self.reduce_single(o, *vals, reduce_ratios=reduce_ratios))
            return ret

        def reduce_single(self, o, *vals, reduce_ratios=None):
            meth = as_varnode(o).flags.data_parallel_reduce_method
            if meth == 'CONCAT':
                return np.concatnate(vals, axis=0)
            elif meth == 'SUM':
                if reduce_ratios is None:
                    return np.mean(vals)

                sum_v, sum_r = 0, 0
                for v, r in zip(vals, reduce_ratios):
                    sum_v += v * r
                    sum_r += r
                return sum_v / sum_r

    def __init__(self, env):
        self._env = env
        self._inputs = None
        self._outputs = None
        self._output_manager = None

        self._extra_outputs = []
        self._extra_kwoutputs = {}
        self._extra_ops = []
        self._extra_kw_modifiers = []

        self.__compiled = False

    @property
    def session(self):
        return self._env.session

    @property
    def flags(self):
        return self._env.flags

    @notnone_property
    def output_manager(self):
        return self._output_manager

    def add_extra_output(self, out):
        assert not self.__compiled
        self._extra_outputs.append(out)

    def add_extra_kwoutput(self, k, v):
        assert not self.__compiled
        self._extra_kwoutputs[k] = v

    def add_extra_op(self, out):
        assert not self.__compiled
        self._extra_ops.append(as_tftensor(out))

    def extend_extra_kw_modifiers(self, modifiers):
        self._extra_kw_modifiers.extend(modifiers)

    @property
    def compiled(self):
        return self.__compiled

    def compile(self, outputs, inputs=None):
        if self.__compiled:
            logger.warn('function {} already compiled'.format(self))

        if len(self._extra_kwoutputs):
            assert isinstance(outputs, (dict, collections.OrderedDict))
            outputs.update(self._extra_kwoutputs)
        if len(self._extra_outputs):
            if isinstance(outputs, __valid_tensor_types__):
                outputs = [outputs]
            outputs.extend(self._extra_outputs)

        self._inputs = inputs
        self._output_manager, self._outputs = Function.OutputManager.make(outputs)
        self._outputs = list(map(as_tftensor, self._outputs))
        self._outputs.extend(self._extra_ops)
        self.__compiled = True

    def __call__(self, *args, output_raw=False, **kwargs):
        if len(args) > 0:
            assert self._inputs is not None
            assert len(self._inputs) == len(args)
            feed_dict = dict(zip(self._inputs, args))
        else:
            feed_dict = kwargs

        for f in self._extra_kw_modifiers:
            f(feed_dict)
        feed_dict = self.canonize_feed_dict(feed_dict)

        outputs = self.session.run(self._outputs, feed_dict=feed_dict)
        if output_raw:
            return outputs
        return self._output_manager.format(outputs)

    def call(self, *args, **kwargs):
        return self(*args, **kwargs)

    def call_args(self, inputs, output_raw=False):
        if isinstance(inputs, dict):
            outputs = self(output_raw=output_raw, **inputs)
        else:
            outputs = self(*inputs, output_raw=output_raw)
        return outputs

    def map(self, iterable, event_spec=None):
        all_outputs = []
        for inputs in iterable:
            outputs = self.call_args(inputs, output_raw=True)
            all_outputs.append(outputs)
        return self._output_manager.reduce_format(self._outputs, all_outputs)

    @staticmethod
    def canonize_feed_dict(feed_dict):
        res = {}
        for k, v in feed_dict.items():
            if type(k) is str and not k.endswith(':0'):
                k += ':0'
            res[k] = v
        return res

