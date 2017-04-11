# -*- coding:utf8 -*-
# File   : trainer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/31/17
# 
# This file is part of TensorArtist

from .env import SimpleTrainerEnv
from .. import summary
from ..graph.env import Env
from ..graph.tfqueue import QueuedInputFunction
from ...core.event import EventManager, register_event, trigger_event
from ...core.utils.meta import assert_instance, notnone_property
from ...core.utils.cache import cached_property

import math
import tensorflow as tf

__all__ = ['TrainerBase', 'SimpleTrainer']


class TrainerBase(object):
    def __init__(self, nr_iters, env=None, data_provider=None, desc=None):
        self._nr_iters = nr_iters
        self._env = env or SimpleTrainerEnv()
        self._data_provider = data_provider
        self._stop_signal = False
        self._desc = desc
        self._need_feed = True

        assert_instance(self._env, Env)

    @property
    def env(self):
        return self._env

    @property
    def network(self):
        return self._env.network

    @property
    def optimizer(self):
        return self._env.optimizer

    @property
    def desc(self):
        return self._desc

    @notnone_property
    def data_provider(self):
        return self._data_provider

    def set_data_provider(self, data_provider):
        self._data_provider = data_provider
        return self

    @property
    def runtime(self):
        return self.env.runtime

    @property
    def iter(self):
        return self.runtime.get('iter', 0)

    @property
    def epoch_size(self):
        return self.runtime.get('epoch_size', 1)

    def set_epoch_size(self, es):
        self.runtime['epoch_size'] = es
        return self

    @property
    def epoch(self):
        return self.iter // self.epoch_size

    @property
    def nr_iters(self):
        return self._nr_iters

    @property
    def nr_epochs(self):
        return int(math.ceil(self.nr_iters / self.epoch_size))

    @property
    def stop_signal(self):
        return self._stop_signal

    def stop(self):
        self._stop_signal = True

    @cached_property
    def _iter_train(self):
        return iter(self.data_provider(self.env))

    def initialize(self):
        self.env.initialize_all_variables()

    def finalize(self):
        pass

    def register_event(self, name, callback, *args, priority=EventManager.DEF_PRIORITY, **kwargs):
        register_event(self, name, callback, *args, priority=priority, **kwargs)
        return self

    def trigger_event(self, name, *args, **kwargs):
        trigger_event(self, name, self, *args, **kwargs)
        return self

    def dump_snapshot(self):
        self.trigger_event('snapshot:dump:before')
        snapshot = self._dump_snapshot()
        self.trigger_event('snapshot:dump:after', snapshot)
        return snapshot

    def load_snapshot(self, snapshot):
        self.trigger_event('snapshot:load:before', snapshot)
        self._load_snapshot(snapshot)
        self.trigger_event('snapshot:load:after')

    def train(self):
        self.trigger_event('initialization:before')
        self.initialize()
        self.trigger_event('initialization:after')
        self.runtime.setdefault('iter', 0)

        self.trigger_event('optimization:before')

        while self.runtime['iter'] <= self.nr_iters and not self.stop_signal:
            if self.runtime['iter'] == 0:
                inp, out = {}, {}
                self.trigger_event('epoch:before')
                self.trigger_event('iter:before', inp)
                self.trigger_event('iter:after', inp, out)
                self.trigger_event('epoch:after')
            else:
                if self.runtime['iter'] % self.epoch_size == 1:
                    self.trigger_event('epoch:before')

                inp = next(self._iter_train) if self._need_feed else {}
                self.trigger_event('iter:before', inp)
                out = self._run_step(inp)
                self.trigger_event('iter:after', inp, out)

                if self.runtime['iter'] % self.epoch_size == 0:
                    self.trigger_event('epoch:after')

            self.runtime['iter'] += 1

        self.trigger_event('optimization:after')

        self.trigger_event('finalization:begin')
        self.finalize()
        self.trigger_event('finalization:after')

    def _run_step(self, data):
        raise NotImplementedError()

    def _dump_snapshot(self):
        return self.env.dump_snapshot()

    def _load_snapshot(self, snapshot):
        self.env.load_snapshot(snapshot)
        return self


class SimpleTrainer(TrainerBase):
    _fn_train = None

    def initialize(self):
        self._fn_train = self.env.make_optimizable_func(self.network.loss)
        if isinstance(self._fn_train, QueuedInputFunction):
            self._need_feed = False
        super().initialize()

    @notnone_property
    def fn_train(self):
        return self._fn_train

    def _compile_fn_train(self):
        if not self._fn_train.compiled:
            summaries = self.network.merged_summaries
            if summaries is not None:
                self._fn_train.add_extra_kwoutput('summaries', summaries)
            self._fn_train.compile({'loss': self.network.loss})
            if isinstance(self._fn_train, QueuedInputFunction):
                self._fn_train.serve(self.data_provider(self.env))

    def _run_step(self, data):
        self._compile_fn_train()
        out = self._fn_train.call_args(data)
        self.runtime['loss'] = out['loss']
        if 'summaries' in out:
            summaries = tf.Summary.FromString(out['summaries'])
            self.runtime['summaries'] = summaries
        return out
