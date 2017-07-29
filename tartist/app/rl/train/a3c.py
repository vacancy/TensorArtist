# -*- coding:utf8 -*-
# File   : a3c.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/19/17
# 
# This file is part of TensorArtist.

from tartist.core import EnvBox, get_env, get_logger
from tartist.data.rflow.query_pipe import QueryReqPipe, QueryRepPipe
from tartist.nn.graph import reuse_context, Env
from tartist.nn.train import SimpleTrainerEnv, SimpleTrainer

import queue
import threading

logger = get_logger(__file__)

__all__ = ['A3CMaster', 'A3CTrainerEnv', 'A3CTrainer']


class A3CMaster(object):
    on_data_func = None
    on_stat_func = None
    player_func = None
    predictor_func = None

    def __init__(self, env, name, nr_predictors):
        self.name = name
        self.env = env
        self.router = QueryRepPipe(name + '-master', send_qsize=12, mode='tcp')
        self.queue = queue.Queue()

        self._nr_predictors = nr_predictors
        self._players = []
        self._predictors = []

    def _on_data_func(self, router, identifier, inp_data):
        self.on_data_func(self.env, identifier, inp_data)

    def _on_stat_func(self, router, identifier, inp_data):
        if self.on_stat_func:
            self.on_stat_func(self.env, identifier, inp_data)

    def _make_predictor_thread(self, i, func, daemon=True):
        return threading.Thread(target=self.predictor_func, daemon=daemon,
                                args=(i, self.router, self.queue, func))

    def _make_player_proc(self, i, req, daemon=True):
        return EnvBox(target=self.player_func, args=(i, req), daemon=daemon)

    def initialize(self):
        self.router.dispatcher.register('data', self._on_data_func)
        self.router.dispatcher.register('stat', self._on_stat_func)
        self.router.initialize()

        assert self._nr_predictors == len(self.env.net_funcs)

        for i in range(self._nr_predictors):
            func = self.env.net_funcs[i]
            prc = self._make_predictor_thread(i, func, daemon=True)
            self._predictors.append(prc)
        for p in self._predictors:
            p.start()

    def start(self, nr_players, name=None, daemon=True):
        name = name or self.name
        self._players = []
        for i in range(nr_players):
            req = QueryReqPipe(name + ('-%d' % i), self.router.conn_info)
            prc = self._make_player_proc(i, req, daemon=daemon)
            self._players.append(prc)
        for p in self._players:
            p.start()
        if not daemon:
            for p in self._players:
                p.join()

    def finalize(self):
        self.router.finalize()


class A3CTrainerEnv(SimpleTrainerEnv):
    _player_master = None
    _net_funcs = None
    _inference_player_master = None
    _data_queue = None

    owner_trainer = None
    network_maker = None

    @property
    def net_funcs(self):
        return self._net_funcs

    @property
    def player_master(self):
        return self._player_master

    @property
    def inference_player_master(self):
        return self._inference_player_master

    @property
    def data_queue(self):
        return self._data_queue

    def initialize_a3c(self):
        nr_predictors = get_env('a3c.nr_predictors')

        # making net funcs
        self._net_funcs = []
        all_devices = self.slave_devices
        if len(all_devices) == 0:
            all_devices = self.all_devices
        for i in range(nr_predictors):
            dev = all_devices[i % len(all_devices)]
            func = self._make_predictor_net_func(i, dev)
            self._net_funcs.append(func)

        self._initialize_a3c_master()
        self._data_queue = queue.Queue(get_env('trainer.batch_size') * get_env('a3c.data_queue_length_factor', 16))

    def _initialize_a3c_master(self):
        nr_predictors = get_env('a3c.nr_predictors')
        self._player_master = A3CMaster(self, 'a3c-player', nr_predictors)

        self._inference_player_master = A3CMaster(self, 'a3c-inference-player', nr_predictors)

    def initialize_all_peers(self):
        nr_players = get_env('a3c.nr_players')

        self._player_master.initialize()
        if self._inference_player_master is not None:
            self._inference_player_master.initialize()

        # Must call initialize_all_variables before start any players.
        self.initialize_all_variables()
        self._player_master.start(nr_players, daemon=True)

    def finalize_all_peers(self):
        self.player_master.finalize()
        if self._inference_player_master is not None:
            self.inference_player_master.finalize()

    def _make_predictor_net_func(self, i, dev):
        def prefix_adder(feed_dict):
            for k in list(feed_dict.keys()):
                feed_dict['predictor/{}/{}'.format(i, k)] = feed_dict.pop(k)

        outputs_name = get_env('a3c.predictor.outputs_name')
        new_env = Env(master_dev=dev, flags=self.flags, dpflags=self.dpflags, graph=self.graph, session=self.session)
        with new_env.as_default():
            with new_env.name_scope('predictor/{}'.format(i)), reuse_context(True):
                self.network_maker(new_env)
            outs = {k: new_env.network.outputs[k] for k in outputs_name}
            f = new_env.make_func()
            f.extend_extra_kw_modifiers([prefix_adder])
            if f.queue_enabled:
                f.disable_queue()
            f.compile(outputs=outs)
        return f


class A3CTrainer(SimpleTrainer):
    def initialize(self):
        super().initialize()
        self.env.network_maker = self.desc.make_network
        self.env.owner_trainer = self
        self.env.initialize_a3c()
        self.desc.make_a3c_configs(self.env)
        self.env.initialize_all_peers()

    def finalize(self):
        self.env.finalize_all_peers()
        super().finalize()
