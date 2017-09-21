# -*- coding:utf8 -*-
# File   : q_learning.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 21/09/2017
# 
# This file is part of TensorArtist.

from tartist.data.flow import SimpleDataFlowBase
from tartist.random.sampler import EpochBatchSampler
from collections import deque

__all__ = ['QLearningDataFlow']


# TODO:: Prioritized sampling
class QLearningDataFlow(SimpleDataFlowBase):
    _data_keys = ('state', 'action', 'next_state', 'reward', 'is_over')
    _memory = None

    def __init__(self, collector, target, maxsize, batch_size, epoch_size, nr_td_steps=1, gamma=1, reward_cb=None):
        self._collector = collector
        self._target = target
        self._maxsize = maxsize
        self._sampler = EpochBatchSampler(batch_size, epoch_size)

        self._nr_td_steps = nr_td_steps
        self._gamma = gamma
        self._reward_cb = reward_cb

        assert self._nr_td_steps == 1, 'TD mode not implemented.'
        assert self._collector.mode.startswith('EPISODE')

    def _initialize(self):
        self._collector.initialize()
        self._memory = {k: deque(maxlen=self._maxsize) for k in self._data_keys}

    def _gen(self):
        while True:
            data = self._collector.collect(self._target)
            self._add_to_memory(data)
            for batch in self._sampler(self._memory, keys=self._data_keys):
                yield batch

    def _process_reward(self, r):
        if self._reward_cb is None:
            return r
        return self._reward_cb(r)

    def _add_to_memory_step(self, e, f):
        for key in ['state', 'action', 'is_over']:
            self._memory[key].append(getattr(e, key))
        self._memory['next_state'].append(f.state)
        self._memory['reward'].append(self._process_reward(e.reward))

    def _add_to_memory(self, raw_data):
        for t in raw_data:
            for i, (e, f) in enumerate(zip(t[:-1], t[1:])):
                self._add_to_memory_step(e, f)

            if len(t) and t[-1].is_over:
                self._add_to_memory_step(t[-1], t[-1])
