# -*- coding:utf8 -*-
# File   : a3c.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/19/17
# 
# This file is part of TensorArtist

from ...core import EnvBox, get_env, get_logger
from ...core.utils.meta import notnone_property
from ...data.rflow.query_pipe import QueryReqPipe, QueryRepPipe
from ...nn.graph import select_device, reuse_context, Env
from ...nn.train import TrainerBase, TrainerEnv, SimpleTrainer

import queue
import threading
import tensorflow as tf

logger = get_logger(__file__)

__all__ = ['A3CTrainerEnv', 'A3CTrainer']


class A3CMaster(object):
    def __init__(self, env, name, nr_predictors):
        self.name = name
        self.env = env
        self.router = QueryRepPipe(name + '-master', send_qsize=128)
        self.queue = queue.Queue()

        self.on_data_func = None
        self.on_stat_func = None
        self.player_func = None
        self.predictor_func = None

        self._nr_predictors = nr_predictors
        self._players = []
        self._predictors = []

    def _on_data_func(self, router, identifier, inp_data):
        self.on_data_func(self.env, identifier, inp_data)

    def _on_stat_func(self, router, identifier, inp_data):
        if self.on_stat_func:
            self.on_stat_func(self.env, identifier, inp_data)

    def initialize(self):
        self.router.dispatcher.register('data', self._on_data_func)
        self.router.dispatcher.register('stat', self._on_stat_func)
        self.router.initialize()

        assert self._nr_predictors == len(self.env.net_funcs)

        for i in range(self._nr_predictors):
            func = self.env.net_funcs[i]
            prc = threading.Thread(target=self.predictor_func, daemon=True,
                                   args=(i, self.router, self.queue, func))
            self._predictors.append(prc)
        for p in self._predictors:
            p.start()

    def start(self, nr_players, daemon=True):
        self._players = []
        for i in range(nr_players):
            req = QueryReqPipe(self.name + ('-%d' % i), self.router.conn_info)
            prc = EnvBox(target=self.player_func, args=(i, req), daemon=daemon)
            self._players.append(prc)
        for p in self._players:
            p.start()
        if not daemon:
            for p in self._players:
                p.join()

    def finalize(self):
        self.router.finalize()


class A3CTrainerEnv(TrainerEnv):
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

    def initialize_all_peers(self):
        nr_players = get_env('a3c.nr_players')
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

        self._player_master = A3CMaster(self, 'a3c-player', nr_predictors)
        self._inference_player_master = A3CMaster(self, 'a3c-inference-player', nr_predictors)
        self._data_queue = queue.Queue(get_env('trainer.batch_size') * get_env('a3c.data_queue_length_factor', 16))

        self._player_master.initialize()
        self._player_master.start(nr_players, daemon=True)
        self._inference_player_master.initialize()

    def finialize_all_peers(self):
        self.player_master.finalize()

    def _make_predictor_net_func(self, i, dev):
        def prefix_adder(feed_dict):
            for k in list(feed_dict.keys()):
                feed_dict['predictor/{}/{}'.format(i, k)] = feed_dict.pop(k)

        outputs_name = get_env('a3c.predictor.outputs_name')
        new_env = Env(master_dev=dev, flags=self.flags, dpflags=self.dpflags, graph=self.graph, session=self.session)
        with new_env.as_default():
            with tf.name_scope('predictor/{}'.format(i)), reuse_context(True):
                self.network_maker(new_env)
            new_env.initialize_all_variables()
            outs = {k: new_env.network.outputs[k] for k in outputs_name}
            f = new_env.make_func()
            f.extend_extra_kw_modifiers([prefix_adder])
            f.compile(outputs=outs)
        return f


class A3CTrainer(SimpleTrainer):
    def initialize(self):
        super().initialize()
        self.env.network_maker = self.desc.make_network
        self.env.owner_trainer = self
        self.desc.make_a3c_configs(self.env)
        self.env.initialize_all_peers()
