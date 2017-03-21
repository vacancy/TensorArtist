# -*- coding:utf8 -*-
# File   : trainer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/31/17
# 
# This file is part of TensorArtist

from .env import TrainerEnv
from ..graph.env import Env
from ..graph.tfqueue import QueuedInputFunction
from ...core import trigger_event
from ...core.utils.meta import assert_instance, notnone_property
from ...core.utils.cache import cached_property

import math
import tensorflow as tf

__all__ = ['TrainerBase', 'SimpleTrainer']


class TrainerBase(object):
    def __init__(self, nr_iters, env=None, data_provider=None, desc=None):
        self._nr_iters = nr_iters
        self._env = env or TrainerEnv()
        self._data_provider = data_provider
        self._runtime = dict()
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

    def load_snapshot(self, snapshot):
        trigger_event(self, 'snapshot:load:before', self, snapshot)
        self._load_snapshot(snapshot)
        trigger_event(self, 'snapshot:load:after', self)

    def dump_snapshot(self):
        trigger_event(self, 'snapshot:dump:before', self)
        snapshot = self._dump_snapshot()
        trigger_event(self, 'snapshot:dump:after', self, snapshot)
        return snapshot

    def train(self):
        trigger_event(self, 'initialization:before', self)
        self.initialize()
        trigger_event(self, 'initialization:after', self)
        self.runtime.setdefault('iter', 0)

        trigger_event(self, 'optimization:before', self)

        while self.runtime['iter'] <= self.nr_iters and not self.stop_signal:
            if self.runtime['iter'] == 0:
                inp, out = {}, {}
                trigger_event(self, 'epoch:before', self)
                trigger_event(self, 'iter:before', self, inp)
                trigger_event(self, 'iter:after', self, inp, out)
                trigger_event(self, 'epoch:after', self)
            else:
                if self.runtime['iter'] % self.epoch_size == 1:
                    trigger_event(self, 'epoch:before', self)

                inp = next(self._iter_train) if self._need_feed else {}
                trigger_event(self, 'iter:before', self, inp)
                out = self._run_step(inp)
                trigger_event(self, 'iter:after', self, inp, out)

                if self.runtime['iter'] % self.epoch_size == 0:
                    trigger_event(self, 'epoch:after', self)

            self.runtime['iter'] += 1

        trigger_event(self, 'optimization:after', self)

        trigger_event(self, 'finalization:before', self)
        self.finalize()
        trigger_event(self, 'finalization:after', self)

    def _run_step(self, data):
        raise NotImplementedError()

    def _dump_snapshot(self):
        variables = self.network.fetch_all_variables_dict()
        runtime = self.runtime.copy()
        snapshot = dict(variables=variables, runtime=runtime)
        return snapshot

    def _load_snapshot(self, snapshot):
        variables = snapshot['variables']
        runtime = snapshot['runtime'].copy()
        self._runtime = runtime
        self.network.assign_all_variables_dict(variables)


class SimpleTrainer(TrainerBase):
    _fn_train = None

    def initialize(self):
        super().initialize()
        self._fn_train = self.env.make_optimizable_func(self.network.loss)
        if isinstance(self._fn_train, QueuedInputFunction):
            self._need_feed = False

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
