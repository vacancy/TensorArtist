# -*- coding:utf8 -*-
# File   : pcollector.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/07/2017 # 
# This file is part of TensorArtist.

import os
import collections
import itertools
import os.path as osp
import threading
import uuid

import imageio
import numpy as np
import sortedcollections
from tornado import ioloop, template
from tornado.web import Application, StaticFileHandler

from tartist.core import get_env, get_logger
from tartist.core import io
from tartist.core.utils.network import get_local_addr
from .rpredictor import TrainingData

TrajectoryPair = collections.namedtuple('TrajectoryPair', ['t1_state', 't1_observation', 't1_action',
                                                           't2_state', 't2_observation', 't2_action'])
_TrajectoryPairWrapper = collections.namedtuple('_TrajectoryPairWrapper', ['priority', 'count', 'pair'])

logger = get_logger(__file__)


def _compose_dir(uuid):
    dirname = osp.join(get_env('dir.root'), 'trajectories', uuid)
    return dirname

   
def _save_gif(traj, filename):
    traj = np.asarray(traj, dtype='uint8')
    return imageio.mimwrite(filename, traj,  duration=0.1)


class PreferenceCollector(object):
    """
    Preference collector is the interface for prefcol's front end.
    """
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

        self._webserver = WebServer(self, configs=web_configs)
        self._webserver_thread = None

    def ready_for_step(self, epoch):
        return self._rpredictor.ready_for_step(epoch)

    def initialize(self):
        self.__restore_preferences()
        self._webserver_thread = threading.Thread(target=self._webserver.mainloop, daemon=True)
        self._webserver_thread.start()

    def __restore_preferences(self):
        """Restore the preferences we already have."""
        dirname = osp.join(get_env('dir.root'), 'trajectories')

        if not osp.isdir(dirname):
            return

        all_data = []
        logger.critical('Restoring preference')
        for uid in os.listdir(dirname):
            item = osp.join(dirname, uid)
            pref_filename = osp.join(item, 'pref.txt')
            pair_filename = osp.join(item, 'pair.pkl')
            if osp.exists(pref_filename) and osp.exists(pair_filename):
                pref = float(io.load(pref_filename)[0])
                pair = io.load(pair_filename)

                data = TrainingData(pair.t1_state, pair.t1_action, pair.t2_state, pair.t2_action, pref)
                all_data.append(data)

        if len(all_data) > 0:
            self._rpredictor.extend_training_data(all_data)

        logger.critical('Preference restore finished: success={}'.format(len(all_data)))

    @property
    def pool(self):
        return self._pool

    def post_state(self, identifier, state, observation, action, variance):
        # logger.info('Post state identifier={}, state={}.'.format(identifier, id(state)))

        data = self._data[identifier]
        data.append((state, observation, action, variance))
        if len(data) == data.maxlen:
            self._try_post_video(data)

    def post_preference(self, uid, pref):
        dirname = _compose_dir(uid)
        pair = io.load(osp.join(dirname, 'pair.pkl'))
        io.dump(osp.join(dirname, 'pref.txt'), str(pref))

        logger.info('Post preference uid={}, pref={}.'.format(uid, pref))

        data = TrainingData(pair.t1_state, pair.t1_action, pair.t2_state, pair.t2_action, pref)
        self._rpredictor.add_training_data(data)

    def _try_post_video(self, data):
        current_sum = 0
        for i in range(self._video_length):
            current_sum += data[i][-1]

        max_sum, max_idx = current_sum, 0
        for i in range(self._video_length, self._window_length):
            current_sum = current_sum - data[i-self._video_length][-1] + data[i][-1]
            if current_sum > max_sum:
                max_sum, max_idx = current_sum, i - self._video_length

        this_states, this_observations, this_actions = [], [], []
        for i in range(max_idx):
            data.popleft()
        for i in range(self._video_length):
            d = data.popleft()
            this_states.append(d[0])
            this_observations.append(d[1])
            this_actions.append(d[2])

        # convert the output to ndarrays
        this_states, this_observations, this_actions = map(np.array, (this_states, this_observations, this_actions))

        self._try_post_pair(this_states, this_observations, this_actions, max_sum)

    def _try_post_pair(self, states, observations, actions, variance):
        self._pair_buffer.append((states, observations, actions, variance))

        if len(self._pair_buffer) == 2:
            t1, t2 = self._pair_buffer
            self._pool.push(t1[0], t1[1], t1[2], t2[0], t2[1], t2[2], t1[3] + t2[3])
            self._pair_buffer.clear()


class TrajectoryPairPool(object):
    """
    Trajectory pool is a simple pool contains a set of trajectories. In this implementation, we maintain a priority
    queue of given max size. When new trajectory comes in, we push it into the priority queue and pop out the one
    with least priority.

    When pool.pop() method is called, the trajectory pair with highest priority is poped out. The program generates the
    demonstration file (e.g. GIF animation) and return an UID and the pair.

    See _dump method for details of dump.
    """
    def __init__(self, maxlen=100, tformat='GIF'):
        """
        Initialize an empty trajectory pair pool.
        :param maxlen: max size of the pool.
        :param tformat: trajectory format, default "GIF".
        """
        self._data_pool = sortedcollections.SortedList()

        self._data_pool_counter = itertools.count()
        self._data_pool_lock = threading.Lock()

        self._maxlen = maxlen
        self._tformat = tformat

    def push(self, t1_state, t1_observation, t1_action, t2_state, t2_observation, t2_action, priority):

        # logger.info('Got pushed trajectory: len1={}, len2={}, priority={}.'.format(
        #     len(t1_state), len(t2_state), priority
        # ))

        with self._data_pool_lock:
            wrapped = _TrajectoryPairWrapper(priority=priority, count=next(self._data_pool_counter),
                                             pair=TrajectoryPair(
                                                 t1_state, t1_observation, t1_action,
                                                 t2_state, t2_observation, t2_action))
            self._data_pool.add(wrapped)
            if len(self._data_pool) == self._maxlen:
                self._data_pool.pop(0)

    def pop(self):
        with self._data_pool_lock:
            if len(self._data_pool) == 0:
                return None, None
            return self._process(self._data_pool.pop().pair)

    def _process(self, pair):
        uid = uuid.uuid4().hex
        dirname = _compose_dir(uid)
        io.mkdir(dirname)
       
        # dump the file for displaying
        if self._tformat == 'GIF':
            _save_gif(pair.t1_observation, osp.join(dirname, '1.gif'))
            _save_gif(pair.t2_observation, osp.join(dirname, '2.gif'))
        else:
            raise ValueError('Unknown trajectory format: {}'.format(self._tformat))

        # cleanup
        pair = TrajectoryPair(pair.t1_state, None, pair.t1_action, pair.t2_state, None, pair.t2_action)
        # dump the raw pair
        io.dump(osp.join(dirname, 'pair.pkl'), pair)

        return uid, pair


class WebServer(object):
    def __init__(self, collector, configs):
        from . import _web_handlers as handlers

        self._template_loader = template.Loader(osp.join(osp.dirname(__file__), '_tpl'))
        self._configs = configs

        init_kwargs = dict(collector=collector, loader=self._template_loader, configs=configs)
        self._application = Application([
            (r'/', handlers.MainHandler, init_kwargs),
            (r'/get', handlers.GetHandler, init_kwargs),
            (r'/submit', handlers.SubmitHandler, init_kwargs),
            (r'/trajectories/(.*)', StaticFileHandler, {'path': osp.join(get_env('dir.root'), 'trajectories')})
        ], debug=True)

    @property
    def application(self):
        return self._application

    @property
    def port(self):
        return self._configs['port']

    def mainloop(self):
        logger.critical('Opening web server at: http://{}:{} .'.format(get_local_addr(), self.port))
        self._application.listen(self.port)
        ioloop.IOLoop.current().start()
