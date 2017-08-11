# -*- coding:utf8 -*-
# File   : push_pipe.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/2/17
# 
# This file is part of TensorArtist.


from . import configs, utils
from ...core.utils.meta import notnone_property

import zmq
import threading
import queue
import contextlib
import collections
import pickle
import functools
# import msgpack
# import msgpack_numpy
# msgpack_numpy.patch()
# dumpb = functools.partial(msgpack.dumps, use_bin_type=True)
# loadb = msgpack.loads

import pickle
dumpb = pickle.dumps
loadb = pickle.loads

__all__ = ['PushPipe', 'PullPipe', 'make_push_pair']


class PullPipe(object):
    def __init__(self, name, mode='tcp'):
        self._name = name
        self._mode = mode
        self._conn_info = None

        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.PULL)
        self._sock.set_hwm(2)

    @notnone_property
    def conn_info(self):
        return self._conn_info

    def initialize(self):
        if self._conn_info is not None:
            return

        if self._mode == 'tcp':
            port = self._sock.bind_to_random_port('tcp://*')
            self._conn_info = 'tcp://{}:{}'.format(utils.get_addr(), port)
        elif self._mode == 'ipc':
            self._conn_info = utils.bind_to_random_ipc(self._sock, self._name)

    def finalize(self):
        utils.graceful_close(self._sock)
        self._context.term()

    @contextlib.contextmanager
    def activate(self):
        self.initialize()
        try:
            yield
        finally:
            self.finalize()
    
    def recv(self):
        try:
            return loadb(self._sock.recv(copy=False).bytes)
        except zmq.ContextTerminated:
            pass


class PushPipe(object):
    def __init__(self, conn_info, send_qsize=10):
        self._conn_info = conn_info
        self._send_qsize = send_qsize

        self._context = None
        self._sock = None
        self._send_queue = None
        self._send_thread = None

    def initialize(self):
        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.PUSH)
        self._sock.set_hwm(2)
        self._sock.connect(self._conn_info)
        self._send_queue = queue.Queue(maxsize=self._send_qsize)

        self._send_thread = threading.Thread(target=self.mainloop_send, daemon=True)
        self._send_thread.start()
        
    def finalize(self):
        utils.graceful_close(self._sock)
        self._context.term()

    @contextlib.contextmanager
    def activate(self):
        self.initialize()
        try:
            yield
        finally:
            self.finalize()
    
    def mainloop_send(self):
        try:
            while True:
                job = self._send_queue.get()
                self._sock.send(dumpb(job), copy=False)
        except zmq.ContextTerminated:
            pass

    def send(self, payload):
        self._send_queue.put(payload)
        return self 


def make_push_pair(name, nr_workers=None, mode='tcp', send_qsize=10):
    pull = PullPipe(name, mode=mode)
    pull.initialize()
    nr_pushs = nr_workers or 1
    pushs = [PushPipe(pull.conn_info, send_qsize=send_qsize) for i in range(nr_pushs)]

    if nr_workers is None:
        return pull, pushs[0]
    return pull, pushs
