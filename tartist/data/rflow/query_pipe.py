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

QueryMessage = collections.namedtuple('QueryMessage', ['identifier', 'payload', 'countdown'])
router_recv = functools.partial(utils.router_recv_json, loader=pickle.loads)
router_send = functools.partial(utils.router_send_json, dumper=pickle.dumps)
req_recv = functools.partial(utils.req_recv_json, loader=pickle.loads)
req_send = functools.partial(utils.req_send_json, dumper=pickle.dumps)


class QueryRepPipe(object):
    def __init__(self, name):
        self._name = name
        self._conn_info = None

        self._context_lock = threading.Lock()
        self._context = zmq.Context()
        self._router = self._context.socket(zmq.ROUTER)
        self._dealer = self._context.dealer(zmq.DEALER)
        self._dispatcher = CallbackManager()

        self._send_queue = queue.Queue()
        self._thread = None
        self._stop_event = threading.Event()

    @property
    def dispatcher(self):
        return self._dispatcher

    @notnone_property
    def conn_info(self):
        return self._conn_info

    def initialize(self):
        self._conn_info = []
        port = self._router.bind_to_random_port('tcp://*')
        self._conn_info.append('tcp://{}:{}'.format(utils.get_addr(), port))
        port = self._dealer.bind_to_random_port('tcp://*')
        self._conn_info.append('tcp://{}:{}'.format(utils.get_addr(), port))

        self._thread = threading.Thread(target=self.mainloop)
        self._thread.start()

    def finalize(self):
        self._stop_event.set()
        self._thread.join()
        utils.graceful_close(self._router)

    @contextlib.contextmanager
    def activate(self):
        self.initialize()
        try:
            yield
        finally:
            self.finalize()

    def mainloop(self):
        while True:
            if self._stop_event.is_set():
                break
            
            nr_done = 0
            nr_done += self._main_do_send()
            nr_done += self._main_do_recv()

    def _main_do_send(self):
        nr_send = self._send_queue.qsize()
        nr_done = 0

        for i in range(nr_send):
            job = self._send_queue.get()
            rc = router_send(self._dealer, job.identifier, job.payload, flag=zmq.NOBLOCK)
            if not rc:
                if job.countdown > 0:
                    self._send_queue.put(QueryMessage(job[0], job[1], job[2] - 1))
            else:
                nr_done += 1
        return nr_done

    def _main_do_recv(self):
        nr_done = 0
        for identifier, msg in utils.iter_recv(router_recv, self._router):
            self._dispatcher.dispatch(msg['type'], self, identifier, msg)
            nr_done += 1
        return nr_done

    def send(self, identifier, msg):
        self._send_queue.put(QueryMessage(identifier, msg, configs.QUERY_REP_COUNTDOWN))


class QueryReqPipe(object):
    def __init__(self, name, conn_info):
        self._name = name
        self._conn_info = conn_info
        self._context = None
        self._push = None
        self._pull = None

    def initialize(self):
        self._context = zmq.Context()
        self._push = self._context.socket(zmq.PUSH)
        self._pull = self._context.socket(zmq.PULL)
        self._push.connect(self._conn_info[0])
        self._pull.connect(self._conn_info[1])

    def finalize(self):
        utils.graceful_close(self._push)
        utils.graceful_close(self._pull)

    @contextlib.contextmanager
    def activate(self):
        self.initialize()
        try:
            yield
        finally:
            self.finalize()

    def query(self, inp):
        req_send(self._push, inp)
        out = req_recv(self._pull)
        return out

