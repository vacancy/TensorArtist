# -*- coding:utf8 -*-
# File   : base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
# 
# This file is part of TensorArtist

from ...core.logger import get_logger
import collections
logger = get_logger(__file__)

__all__ = ['DataFlowBase', 'SimpleDataFlowBase', 'AdvancedDataFlowBase']


class DataFlowBase(object):
    pass

collections.Iterable.register(DataFlowBase)


class SimpleDataFlowBase(DataFlowBase):
    __initialized = False

    def _initialize(self):
        pass

    def _reset(self):
        pass
    
    def _gen(self):
        raise NotImplementedError()

    def _finalize(self):
        pass

    def _len(self):
        return None

    def __len__(self):
        try:
            return self._len()
        except TypeError:
            return None

    def __iter__(self):
        if not self.__initialized:
            self._initialize()
            self.__initialized = True 
        self._reset()
        try:
            for v in self._gen():
                yield v
        except Exception as e:
            logger.warn('{} got exception {} during iter: {}'.format(type(self), type(e), e))
            pass
        finally:
            self._finalize()


class AdvancedDataFlowBase(DataFlowBase):
    def __init__(self):
        self._is_first_iter = True

    def __len__(self):
        return self._count()

    def __iter__(self):
        self._initialize()
        self._is_first_iter = True
        return self

    def __next__(self):
        if not self._is_first_iter:
            if self._have_next():
                self._move_next()
            else:
                self._finalize()
                raise StopIteration()
        else:
            self._is_first_iter = False
        result = self._get()
        return result

    def _initialize(self):
        raise NotImplementedError()

    def _finalize(self):
        pass

    def _get(self):
        raise NotImplementedError()

    def _count(self):
        raise NotImplementedError()

    def _move_next(self):
        raise NotImplementedError()

    def _have_next(self):
        raise NotImplementedError()

