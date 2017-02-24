# -*- coding:utf8 -*-
# File   : batch.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
# 
# This file is part of TensorArtist

from .base import SimpleDataFlowBase
from ...core.utils.concurrent import MTBooleanEvent
from ...core.utils.meta import iter_kv, assert_none

from copy import copy, deepcopy
from threading import Thread, Event

__all__ = ['BatchDataFlow']


def batch_default_filler(buffer, idx, val):
    for k, v in iter_kv(val):
        buffer[k][idx] = v


class BatchDataFlow(SimpleDataFlowBase):
    _buffer = None
    _cond = None

    _filler_thread = None
    _stop_event = None

    def __init__(self, source, batch_size, sample_dict, filler=batch_default_filler):
        super().__init__()
        self._source = source
        self._batch_size = batch_size
        self._sample_dict = sample_dict
        self._filler = filler

    def _initialize(self):
        self._initialize_buffer()
        self._initialize_filler()

    def _initialize_buffer(self):
        self._buffer = [deepcopy(self._sample_dict) for _ in range(2)]

    def _initialize_filler(self):
        self._cond = [MTBooleanEvent() for _ in range(2)]
        self._stop_event = Event()
        self._filler_thread = Thread(target=self._filler_mainloop, name=str(self) + ':filler', daemon=True)
        self._filler_thread.start()

    def _filler_mainloop(self):
        current = 0
        it = iter(self._source)
        try:
            while True:
                self._cond[current].wait_false()
                for i in range(self._batch_size):
                    self._filler(self._buffer[current], i, next(it))
                self._cond[current].set_true()
                current = 1 - current
        except Exception as e:
            print(type(e), e)
            self._cond[current].set_true()
            self._stop_event.set()

    def _gen(self):
        current = 0
        while True:
            self._cond[current].wait_true()
            if self._stop_event.is_set():
                return
            yield self._buffer[current]
            self._cond[current].set_false()
            current = 1 - current

