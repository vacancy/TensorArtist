# -*- coding:utf8 -*-
# File   : concurrent_monitor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 6/21/17
# 
# This file is part of TensorArtist.

import itertools
import threading
import collections
import time

__all__ = ['TSCounter', 'TSCounterMonitor']


class TSCounter(object):
    def __init__(self):
        self._cnt = itertools.count()
        self._ref = itertools.count()
        self._iter_cnt = iter(self._cnt)
        self._iter_ref = iter(self._ref)

    def tick(self):
        next(self._iter_cnt)

    def get(self):
        ref = next(self._iter_ref)
        cnt = next(self._iter_cnt)
        return cnt - ref
    

class TSCounterMonitor(object):
    _displayer = None

    def __init__(self, counters=None, display_names=None, interval=1, printf=None):
        if counters is None:
            counters = ['DEFAULT']

        self._display_names = display_names
        self._counters = collections.OrderedDict([(n, TSCounter()) for n in counters])
        self._interval = interval
        self._printf = printf

        if self._printf is None:
            from ..logger import get_logger
            logger = get_logger(__file__)
            self._printf = logger.info
    
    @property
    def _counter_names(self):
        return list(self._counters.keys())

    def tick(self, name=None):
        if len(self._counter_names) == 1:
            self._counters[self._counter_names[0]].tick()
        else:
            assert name is None, 'Must provide name if there are multiple counters.'
            self._counters[name].tick()

    def start(self):
        self._displayer = threading.Thread(target=self._display_thread, daemon=True)
        self._displayer.start()
        return self

    def _display(self, deltas, interval):
        names = self._display_names or self._counter_names
        if len(names) == 1:
            self._printf('Counter monitor {}: {} ticks/s.'.format(names[0], deltas[0]/interval))
        else:
            log_strs = ['Counter monitor:']
            for n, v in zip(names, deltas):
                log_strs.append('\t{}: {} ticks/s'.format(n, v/interval))
            self._printf('\n'.join(log_strs))

    def _display_thread(self):
        prev = [c.get() for _, c in self._counters.items()]
        while True:
            time.sleep(self._interval)
            curr = [c.get() for _, c in self._counters.items()]
            deltas = [c - p for p, c in zip(prev, curr)]
            prev = curr
            self._display(deltas, self._interval)
