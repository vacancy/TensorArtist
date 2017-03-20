# -*- coding:utf8 -*-
# File   : query_pipe.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/19/17
# 
# This file is part of TensorArtist


from . import configs, utils
from ...core.utils.callback import CallbackManager
from ...core.utils.meta import notnone_property

import zmq
import threading
import queue
import contextlib
import collections
import pickle
import functools

QueryMessage = collections.namedtuple('QueryMessage', ['identifier', 'payload'])
dumpb = pickle.dumps
loadb = pickle.loads


class QueryRepPipe(object):
    def __init__(self, name):
        self._name = name
        self._conn_info = None

        self._context_lock = threading.Lock()
        self._context = zmq.Context()
        self._tosock = self._context.socket(zmq.ROUTER)
        self._frsock = self._context.socket(zmq.PULL)
        self._tosock.set_hwm(10)
        self._frsock.set_hwm(10)
        self._dispatcher = CallbackManager()

        self._send_queue = queue.Queue()
        self._rcv_thread = None
        self._snd_thread = None

    @property
    def dispatcher(self):
        return self._dispatcher

    @notnone_property
    def conn_info(self):
        return self._conn_info

    def initialize(self):
        self._conn_info = []
        # port = self._frsock.bind_to_random_port('tcp://*')
        # self._conn_info.append('tcp://{}:{}'.format(utils.get_addr(), port))
        # port = self._tosock.bind_to_random_port('tcp://*')
        # self._conn_info.append('tcp://{}:{}'.format(utils.get_addr(), port))
        self._conn_info.append(utils.bind_to_random_ipc(self._frsock, self._name + '-c2s-'))
        self._conn_info.append(utils.bind_to_random_ipc(self._tosock, self._name + '-s2c-'))

        self._rcv_thread = threading.Thread(target=self.mainloop_recv)
        self._rcv_thread.start()
        self._snd_thread = threading.Thread(target=self.mainloop_send)
        self._snd_thread.start()

    def finalize(self):
        utils.graceful_close(self._tosock)
        utils.graceful_close(self._frsock)
        self._context.term()

    @contextlib.contextmanager
    def activate(self):
        self.initialize()
        try:
            yield
        finally:
            self.finalize()

    def mainloop_recv(self):
        try:
            while True:
                msg = loadb(self._frsock.recv(copy=False).bytes)
                identifier, type, payload = msg
                self._dispatcher.dispatch(type, self, identifier, payload)
        except zmq.ContextTerminated:
            pass

    def mainloop_send(self):
        try:
            while True:
                job = self._send_queue.get()
                self._tosock.send_multipart((job.identifier, b'', dumpb(job.payload)), copy=False)
        except zmq.ContextTerminated:
            pass

    def send(self, identifier, msg):
        self._send_queue.put(QueryMessage(identifier, msg))


class QueryReqPipe(object):
    def __init__(self, name, conn_info):
        self._name = name
        self._conn_info = conn_info
        self._context = None
        self._tosock = None
        self._frsock = None

    @property
    def identity(self):
        return self._name.encode('utf-8')

    def initialize(self):
        self._context = zmq.Context()
        self._tosock = self._context.socket(zmq.PUSH)
        self._frsock = self._context.socket(zmq.DEALER)
        self._tosock.setsockopt(zmq.IDENTITY, self.identity)
        self._frsock.setsockopt(zmq.IDENTITY, self.identity)
        self._tosock.set_hwm(2)
        self._tosock.connect(self._conn_info[0])
        self._frsock.connect(self._conn_info[1])

    def finalize(self):
        utils.graceful_close(self._frsock)
        utils.graceful_close(self._tosock)
        self._context.term()

    @contextlib.contextmanager
    def activate(self):
        self.initialize()
        try:
            yield
        finally:
            self.finalize()

    def query(self, type, inp, do_recv=True):
        self._tosock.send(dumpb((self.identity, type, inp)))
        if do_recv:
            out = loadb(self._frsock.recv_multipart(copy=False)[1].bytes)
            return out
