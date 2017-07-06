# -*- coding:utf8 -*-
# File   : webserver.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/07/2017
# 
# This file is part of TensorArtist.

from tornado.web import Application, RequestHandler
from tornado import ioloop, template
import random
import os.path as osp
import heapq
import json
import threading
import itertools
import collections


TrajectoryPair = collections.namedtuple('TrajectoryPair', ['t1', 't2'])
_TrajectoryPairWrapper = collections.namedtuple('_TrajectoryPairWrapper', ['priority', 'count', 'pair'])


class TrajectoryPairPool(object):
    def __init__(self, maxlen=100):
        self._data_pool = []
        self._data_pool_counter = itertools.count()
        self._data_pool_lock = threading.Lock()
        self._maxlen = maxlen

    def push(self, t1, t2, priority):
        with self._data_pool_lock:
            wrapped = _TrajectoryPairWrapper(priority=-priority, count=next(self._data_pool_counter),
                                             pair=TrajectoryPair(t1, t2))
            if len(self._data_pool) == self._maxlen:
                heapq.heapreplace(self._data_pool, wrapped)
            else:
                heapq.heappush(self._data_pool, wrapped)

    def pop(self):
        with self._data_pool_lock:
            return heapq.heappop(self._data_pool).pair


class _HPHandlerBase(RequestHandler):
    _pool = None
    _loader = None
    _configs = None

    def initialize(self, pool=None, loader=None, configs=None):
        self._pool = pool
        self._loader = loader
        self._configs = configs


class _MainHandler(_HPHandlerBase):
    def get(self):
        self.write(self._loader.load('index.html').generate(
            site_title=self._configs['title'],
            site_author=self._configs['author'],
        ))


class _GetHandler(_HPHandlerBase):
    def get(self):
        import time; time.sleep(2)
        self.write(json.dumps({
            'id': random.randint(1, 32767),
            'traj1': '<div class="hp-placeholder"></div>',
            'traj2': '<div class="hp-placeholder"></div>'
        }))


class _SubmitHandler(_HPHandlerBase):
    def post(self):
        self.write('Received: id={}, pref={}.'.format(self.get_argument('id'), self.get_argument('pref')))


class WebServer(object):
    def __init__(self, pool, configs):
        self._template_loader = template.Loader(osp.join(osp.dirname(__file__), '_tpl'))

        kwargs = dict(pool=pool, loader=self._template_loader, configs=configs)
        self._application = Application([
            (r'/', _MainHandler, kwargs),
            (r'/get', _GetHandler, kwargs),
            (r'/submit', _SubmitHandler, kwargs)
        ], debug=True)
        self._pool = pool
        self._configs = configs

    @property
    def application(self):
        return self._application

    @property
    def port(self):
        return self._configs['port']

    def mainloop(self):
        self._application.listen(self.port)
        ioloop.IOLoop.current().start()
