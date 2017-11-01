# -*- coding:utf8 -*-
# File   : remote.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/10/17
# 
# This file is part of TensorArtist.

from .base import SimpleDataFlowBase
from ..rflow import InputPipe, OutputPipe, control
from ..rflow import make_push_pair
from ...random import gen_seed, reset_global_rng

from multiprocessing import Process
import time

__all__ = ['RemoteDataFlow', 'MPPrefetchDataFlow', 'MPCustomDataFlow', 'RemoteMonitorDataFlow']


class RemoteDataFlow(SimpleDataFlowBase):
    def __init__(self, pipe_name, bufsize=100):
        self._pipe = InputPipe(pipe_name, bufsize=bufsize)

    def _gen(self):
        with control([self._pipe]):
            while True:
                yield self._pipe.get()


class MPPrefetchDataFlow(SimpleDataFlowBase):
    def _mainloop_worker(self, wid, seed):
        reset_global_rng(seed)
        with self._pushs[wid].activate():
            for data in self._dataflow:
                self._pushs[wid].send(data)

    def __init__(self, dataflow, nr_workers=1, mode='tcp', send_qsize=10):
        self._dataflow = dataflow
        self._nr_workers = nr_workers
        self._mode = mode
        self._send_qsize = send_qsize
        self._pull = None
        self._pushs = None
        self._procs = None

    def _initialize(self):
        super()._initialize()
        self._pull, self._pushs = make_push_pair(str(self), self._nr_workers, mode=self._mode, send_qsize=self._send_qsize)
        self._procs = [Process(target=self._mainloop_worker, args=(i, gen_seed()), daemon=True) for i in range(self._nr_workers)]
        for p in self._procs:
            p.start()
    
    def _gen(self):
        with self._pull.activate():
            while True:
                yield self._pull.recv()


class MPCustomDataFlow(SimpleDataFlowBase):
    def __init__(self, target=None, nr_workers=2, mode='tcp', send_qsize=10):
        self._nr_workers = nr_workers
        self._mode = mode
        self._send_qsize = send_qsize
        self._pull = None
        self._pushs = None

    def run(self, wid, pipe, seed):
        reset_global_rng(seed)
        return self.target(wid, pipe)
 
    def _initialize(self):
        super()._initialize()
        self._pull, self._pushs = make_push_pair(str(self), self._nr_workers, mode=self._mode, send_qsize=self._send_qsize)
        self._procs = [Process(target=self.run, args=(i, self._pushs[i], gen_seed()), daemon=True) for i in range(self._nr_workers)]
        for p in self._procs:
            p.start()
    
    def _gen(self):
        with self._pull.activate():
            while True:
                yield self._pull.recv()


class RemoteMonitorDataFlow(SimpleDataFlowBase):
    def __init__(self, df, pipe_name, bufsize=1):
        self._df = df
        self._pipe = OutputPipe(pipe_name, bufsize=bufsize)

    def _gen(self):
        with control([self._pipe]):
            for data in self._df:
                self._pipe.put_nowait({'data': data, 'time': time.time()})
                yield data
