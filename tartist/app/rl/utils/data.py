# -*- coding:utf8 -*-
# File   : data.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 17/08/2017
# 
# This file is part of TensorArtist.

import numpy as np

from tartist.data.flow import SimpleDataFlowBase
from tartist.random.sampler import SimpleBatchSampler

__all__ = ['SynchronizedTrajectoryDataFlow', 'QLearningDataFlow']


class SynchronizedTrajectoryDataFlow(SimpleDataFlowBase):
    def __init__(self, collector, target, incl_value):
        self._collector = collector
        self._target = target
        self._incl_value = incl_value

        assert self._collector.mode.startswith('EPISODE')

    def _initialize(self):
        self._collector.initialize()

    def _gen(self):
        while True:
            data = self._collector.collect(self._target)
            data = self._process(data)
            yield data

    def _process(self, raw_data):
        data_list = []
        for t in raw_data:
            data = dict(
                step=[],
                state=[],
                action=[],
                theta_old=[],
                reward=[],
                value=[],
                score=0
            )

            for i, e in enumerate(t):
                data['step'].append(i)
                data['state'].append(e.state)
                data['action'].append(e.action)
                data['theta_old'].append(e.outputs['theta'])
                data['reward'].append(e.reward)
                data['score'] += e.reward

                if self._incl_value:
                    data['value'].append(e.outputs['value'])

            if not self._incl_value:
                del data['value']

            for k, v in data.items():
                data[k] = np.array(v)

            if len(t) > 0:
                data_list.append(data)
        return data_list


class QLearningDataFlow(object):
    _data_keys = ('state', 'action', 'q_value')

    def __init__(self, collector, target, gamma, batch_size, nr_repeat, nr_td_steps=1):
        self._collector = collector
        self._target = target
        self._gamma = gamma
        self._nr_td_steps = nr_td_steps
        self._sampler = SimpleBatchSampler(batch_size, nr_repeat)

        assert self._collector.mode.startswith('EPISODE')

    def _initialize(self):
        self._collector.initialize()

    def _gen(self):
        while True:
            data = self._collector.collect(self._target)
            data = self._process(data)
            for batch in self._sampler(data, keys=self._data_keys):
                yield batch

    def _process(self, raw_data):
        data = {k: [] for k in self._data_keys}

        for t in raw_data:
            for i in range(len(t) - self._nr_td_steps, -1, -1):
                e = t[i]

                q = t[i + self._nr_td_steps].outputs['max_q'] if i + self._nr_td_steps < len(t) else 0
                for j in range(self._nr_td_steps - 1, -1, -1):
                    q = q * self._gamma + t[i + j].reward

                data['state'].append(e.state)
                data['action'].append(e.action)
                data['q_value'].append(q)

        for k, v in data.items():
            data[k] = np.array(v)[:self._target]

        return data
