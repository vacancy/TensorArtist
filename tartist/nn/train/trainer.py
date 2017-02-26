# -*- coding:utf8 -*-
# File   : trainer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/31/17
# 
# This file is part of TensorArtist

from .env import TrainerEnv
from ...core import get_env, trigger_event
from ...core.utils.meta import assert_instance, notnone_property
from ...core.utils.cache import cached_property

import tensorflow as tf

__all__ = ['TrainerBase', 'SimpleTrainer']


class TrainerBase(object):
    def __init__(self, env=None, data_provider=None):
        self._env = env or TrainerEnv()
        self._data_provider = data_provider
        self._runtime = dict()
        self._stop_signal = False

        assert_instance(self._env, TrainerEnv)

    @property
    def env(self):
        return self._env

    @property
    def network(self):
        return self._env.network

    @property
    def optimizer(self):
        return self._env.optimizer

    @notnone_property
    def data_provider(self):
        return self._data_provider

    def set_data_provider(self, data_provider):
        self._data_provider = data_provider
        return self

    @property
    def runtime(self):
        return self._runtime

    @property
    def iter(self):
        return self._runtime.get('iter', 0)

    @property
    def epoch_size(self):
        return self._runtime.get('epoch_size', 1)

    def set_epoch_size(self, es):
        self._runtime['epoch_size'] = es
        return self

    @property
    def epoch(self):
        return self.iter // self.epoch_size

    @property
    def stop_signal(self):
        return self._stop_signal

    def stop(self):
        self._stop_signal = True

    @cached_property
    def _iter_train(self):
        return iter(self.data_provider(self.env))

    def _run_step(self, data):
        raise NotImplementedError()

    def initialize(self):
        self.env.initialize_all_variables()

    def finalize(self):
        pass

    def train(self):
        trigger_event('trainer', 'initialize:before', self)
        self.initialize()
        trigger_event('trainer', 'initialize:after', self)
        self.runtime.setdefault('iter', 0)

        trigger_event('trainer', 'optimization:before', self)

        while self.runtime['iter'] <= get_env('trainer.nr_iters') and not self.stop_signal:
            inp, out = next(self._iter_train), {}

            trigger_event('trainer', 'iter:before', self, inp)
            if self.runtime['iter'] != 0:
                out = self._run_step(next(self._iter_train))
            trigger_event('trainer', 'iter:after', self, inp, out)

            self.runtime['iter'] += 1

        trigger_event('trainer', 'optimization:after', self)

        trigger_event('trainer', 'finalize:before', self)
        self.finalize()
        trigger_event('trainer', 'finalize:after', self)


class SimpleTrainer(TrainerBase):
    _fn_train = None

    def initialize(self):
        super().initialize()
        self._fn_train = self.env.make_optimizable_func(self.network.loss)

    @notnone_property
    def fn_train(self):
        return self._fn_train

    def _compile_fn_train(self):
        if not self._fn_train.compiled:
            summaries = self.network.merged_summaries
            if summaries is not None:
                self._fn_train.add_extra_kwoutput('summaries', summaries)
            self._fn_train.compile({'loss': self.network.loss})

    def _run_step(self, data):
        self._compile_fn_train()
        out = self._fn_train.call_args(data)
        self.runtime['loss'] = out['loss']
        if 'summaries' in out:
            summaries = tf.Summary.FromString(out['summaries'])
            self.runtime['summaries'] = summaries
        return out

    # def make_snapshot(self):
    #     snapshot = TrainerSnapshot()
    #     runtime = self.runtime.clone().as_dict()
    #     for k in list(runtime.keys()):
    #         if k.startswith('_'):
    #             del runtime[k]
    #     snapshot['runtime'] = runtime 
    #     snapshot['weights'] = self.network.get_weights()
    #     snapshot['optimizer'] = self._fn_train.optimizer_state.make_checkpoint()
    #     return snapshot

    # def restore_snapshot(self, snapshot):
    #     self.runtime.load(snapshot['runtime'])
    #     self.network.set_weights(snapshot['weights'])
    #     self._fn_train.optimizer_state.restore_checkpoint(snapshot['optimizer'])
    #     self.set_learning_rate()

