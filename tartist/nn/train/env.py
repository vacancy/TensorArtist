# -*- coding:utf8 -*-
# File   : env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
# 
# This file is part of TensorArtist.

from ...core import get_logger
from ...core.event import EventManager, register_event, trigger_event
from ...core.utils.meta import notnone_property
from ..graph.env import Env
from ..graph.node import as_tftensor

logger = get_logger(__file__)

__all__ = ['TrainerEnvBase', 'SimpleTrainerEnv']


class TrainerEnvBase(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshot_parts = dict()
        self._runtime = dict()

        self.add_snapshot_part('variables', self.__dump_network_variable, self.__load_network_variable)
        self.add_snapshot_part('runtime', self.__dump_runtime, self.__load_runtime)

    def __dump_network_variable(self):
        return self.network.fetch_all_variables_dict()

    def __load_network_variable(self, variables):
        self.network.assign_all_variables_dict(variables)

    def __dump_runtime(self):
        return self._runtime.copy()

    def __load_runtime(self, runtime):
        self._runtime = runtime

    @property
    def runtime(self):
        return self._runtime

    def add_snapshot_part(self, identifier, dump, load):
        self._snapshot_parts[identifier] = (dump, load)

    def get_snapshot_parts_ref(self):
        return self._snapshot_parts

    def load_snapshot(self, snapshot):
        for k, v in snapshot.items():
            if k not in self._snapshot_parts:
                logger.warning('Ignored snapshot part: {}'.format(k))
            else:
                loader = self._snapshot_parts[k][1]
                loader(v)
        return self

    def dump_snapshot(self):
        snapshot = dict()
        for identifier, (d, l) in self._snapshot_parts.items():
            snapshot[identifier] = d()
        return snapshot

    def register_event(self, name, callback, *args, priority=EventManager.DEF_PRIORITY, **kwargs):
        register_event(self, name, callback, *args, priority=priority, **kwargs)
        return self

    def trigger_event(self, name, *args, **kwargs):
        trigger_event(self, name, self, *args, **kwargs)
        return self


class SimpleTrainerEnv(TrainerEnvBase):
    _optimizer = None

    @notnone_property
    def optimizer(self):
        return self._optimizer

    def set_optimizer(self, opt):
        self._optimizer = opt
        return self

    def make_optimizable_func(self, loss=None):
        loss = loss or self.network.loss
        loss = as_tftensor(loss)

        func = self.make_func()
        func.add_extra_op(self.optimizer.minimize(loss))
        return func
