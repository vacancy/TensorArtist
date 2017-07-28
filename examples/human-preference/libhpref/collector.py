# -*- coding:utf8 -*-
# File   : collector.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/07/2017
# 
# This file is part of TensorArtist.

import random
import os.path as osp
import heapq
import json
import threading
import itertools
import collections
import uuid
import imageio

from tornado.web import Application, RequestHandler
from tornado import ioloop, template

from tartist.core import io

from .rpredictor import TrainingData

TrajectoryPair = collections.namedtuple('TrajectoryPair', ['t1_state', 't1_action', 't2_state', 't2_action'])
_TrajectoryPairWrapper = collections.namedtuple('_TrajectoryPairWrapper', ['priority', 'count', 'pair'])
 
def _compose_dir(uuid):
    dirname = osp.join(get_env('dir.root'), 'trajectories', uuid)
    return dirname

   
def _save_gif(traj, filename):
    return imageio.mimsave(filename, traj.astype('uint8'), duration=0.1)


class PreferenceCollector(object):
    def __init__(self, rpredictor, web_configs, video_length=100, window_length=300, pool_size=100):
        self._rpredictor = rpredictor
        self._pool = TrajectoryPairPool(maxlen=pool_size)

        self._video_length = video_length
        self._window_length = window_length
        assert self._video_length <= self._window_length

        # `data` holds the working set of each worker (wid => list of observations)
        self._data = collections.defaultdict(lambda: collections.deque(maxlen=window_length))
        # pair buffer is a buffer with size at most 2, used for generating the pair
        self._pair_buffer = []

        self._webserver = WebServer(self._pool, configs=web_configs)

    @property
    def pool(self):
        return self._pool

    def post_state(self, identifier, state, action, variance):
        data = self._data[identifier]
        data.append((state, action, variance))
        if len(data) == data.maxlen:
            self._try_post_video(data)

    def post_preference(self, uid, pref):
        dirname = _compose_dir(uid)
        pair = io.load(osp.join(dirname, 'pair.pkl'))
        io.dump('pref.txt', str(preference))

        data = TrainingData(pair[0], pair[1], pair[2], pair[3], pref)
        self._rpredictor.add_data(data)

    def _try_post_video(self, data):
        current_sum = 0
        for i in range(self._video_length):
            current_sum += data[i][2]

        max_sum, max_idx = current_sum, 0
        for i in range(self._video_length, self._window_length):
            current_sum = current_sum - data[i-self._video_length][2] + data[i][2]
            if current_sum > max_sum:
                max_sum, max_idx = current_sum, i - self._video_length

        this_states = []
        this_actions = []
        for i in range(max_idx):
            data.popleft()
        for i in range(self._video_length):
            d = data.popleft()
            this_states.append(d[0])
            this_actions.append(d[1])

        self._try_post_pair(this_states, this_actions, max_sum)

    def _try_post_pair(self, states, actions, variance):
        self._pair_buffer.append((states, actions, variance))

        if len(self._pair_buffer) == 2:
            t1, t2 = self._pair_buffer
            self._pool.push(t1[0], t1[1], t2[0], t2[1], t1[2] + t2[2])


class TrajectoryPairPool(object):
    def __init__(self, maxlen=100, tformat='GIF'):
        self._data_pool = []
        self._data_pool_counter = itertools.count()
        self._data_pool_lock = threading.Lock()
        self._maxlen = maxlen
        self._tformat = tformat

    def push(self, t1_state, t1_action, t2_state, t2_action, priority):
        with self._data_pool_lock:
            wrapped = _TrajectoryPairWrapper(priority=-priority, count=next(self._data_pool_counter),
                                             pair=TrajectoryPair(t1_state, t1_action, t2_state, t2_action))
            if len(self._data_pool) == self._maxlen:
                heapq.heapreplace(self._data_pool, wrapped)
            else:
                heapq.heappush(self._data_pool, wrapped)

    def pop(self):
        with self._data_pool_lock:
            if len(self._data_pool) == 0:
                return None, None
            return self._process(heapq.heappop(self._data_pool).pair)

    def _process(self, pair):
        uid = uuid.uuid4()
        dirname = _compose_dir(uid)
        io.mkdir(dirname)
       
        # dump the raw pair
        io.dump(osp.join(dirname, 'pair.pkl'), pair)
        # dump the file for displaying
        if self._tformat == 'GIF':
            _save_gif(pair.t1_state, osp.join(dirname, '1.gif'))
            _save_gif(pair.t2_state, osp.join(dirname, '2.gif'))

        return uid, pair


class _HPHandlerBase(RequestHandler):
    _pool = None
    _loader = None
    _configs = None

    def initialize(self, collector =None, loader=None, configs=None):
        self._collector = collector
        self._loader = loader
        self._configs = configs

    @property
    def _pool(self):
        return self._collector.pool


class _MainHandler(_HPHandlerBase):
    def get(self):
        self.write(self._loader.load('index.html').generate(
            site_title=self._configs['title'],
            site_author=self._configs['author'],
        ))


class _GetHandler(_HPHandlerBase):
    def get(self):
        uid, pair = self._pool.pop()

        if uid is not None:
            self.write(json.dumps({
                'rc': 200,
                'id': uid,
                'traj1': '<div class="hp-placeholder"></div>',
                'traj2': '<div class="hp-placeholder"></div>'
            }))
        else:
            self.write(json.dumps({'rc': 404}))


class _SubmitHandler(_HPHandlerBase):
    def post(self):
        uid = self.get_argument('id')
        pref = self.get_argument('pref')
        self.write('Received: id={}, pref={}.'.format(uid, pref))
        self._collector.post_preference(uid, pref)


class WebServer(object):
    def __init__(self, collector, configs):
        self._template_loader = template.Loader(osp.join(osp.dirname(__file__), '_tpl'))

        kwargs = dict(collector=collector, loader=self._template_loader, configs=configs)
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
