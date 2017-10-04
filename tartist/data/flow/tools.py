# -*- coding:utf8 -*-
# File   : tools.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
# 
# This file is part of TensorArtist.

import queue
import threading

from ...core.utils.meta import map_exec
from .base import SimpleDataFlowBase, ProxyDataFlowBase
from itertools import repeat, cycle as cached_cycle
from itertools import takewhile, dropwhile, filterfalse
from itertools import chain
from itertools import starmap
from itertools import islice
from itertools import tee


__all__ = [
    'cycle', 'cycle_n', 'cached_cycle', 'repeat', 'repeat_n',
    'takewhile', 'dropwhile', 'filtertrue', 'filterfalse',
    'chain',
    'map', 'starmap', 'ssmap'
    'islice', 'truncate',
    'tee',
    'MapDataFlow', 'DataFlowMixer'
]

repeat_n = repeat
filtertrue = dropwhile
truncate = islice


# implement cycle self, without any cache
def cycle(iterable, times=None):
    if times is None:
        while True:
            for v in iterable:
                yield v 
    else:
        for i in range(times):
            for v in iterable:
                yield v

cycle_n = cycle


def ssmap(function, iterable):
    for args in iterable:
        yield function(**args)


class MapDataFlow(ProxyDataFlowBase):
    def __init__(self, other, map_func=None):
        super().__init__(other)
        self.__map_func = map_func

    def _map(self, data):
        return self.__map_func(data)

    def _gen(self):
        for data in self.unwrapped:
            yield self._map(data)


class DataFlowMixer(SimpleDataFlowBase):
    def __init__(self, dataflows, buflen=None):
        if buflen is None:
            buflen = len(dataflows)
        self._dataflows = dataflows
        self._queue = queue.Queue(maxsize=buflen)
        self._stop_signal = threading.Event()
        self._comsumers = []

    def _initialize(self):
        self._consumers = [
            threading.Thread(target=self._consumer, args=(ind, df), daemon=True)
            for ind, df in enumerate(self._dataflows)]
        map_exec(threading.Thread.start, self._consumers)

    def _finalize(self):
        self._stop_signal.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        map_exec(threading.Thread.join, self._consumers)

    def _consumer(self, ind, df):
        for data in df:
            self._queue.put(self._wrapper(data, ind))
            if self._stop_signal.is_set():
                break

    def _gen(self):
        while True:
            yield self._queue.get()

    def _wrapper(self, data, ind):
        return (data, ind)
