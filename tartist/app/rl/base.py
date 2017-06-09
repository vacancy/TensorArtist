# -*- coding:utf8 -*-
# File   : base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/18/17
# 
# This file is part of TensorArtist.

from tartist import random
from tartist.core.utils.cache import cached_property
import numpy as np
import collections

__all__ = [
    'RLEnvironBase', 'SimpleRLEnvironBase', 'ProxyRLEnvironBase',
    'ActionSpaceBase', 'DiscreteActionSpace', 'ContinuousActionSpace'
]


class RLEnvironBase(object):
    def __init__(self):
        self._stats = collections.defaultdict(list)

    @property
    def stats(self):
        return self._stats

    def append_stat(self, name, value):
        self._stats[name].append(value)
        return self

    def clear_stats(self):
        self._stats = collections.defaultdict(list)
        return self

    @cached_property
    def action_space(self):
        return self._get_action_space()

    @property
    def current_state(self):
        return self._get_current_state()

    def action(self, action):
        return self._action(action)

    def restart(self, *args, **kwargs):
        return self._restart(*args, **kwargs)

    def finish(self, *args, **kwargs):
        return self._finish(*args, **kwargs)

    def play_one_episode(self, func, ret_states=False, restart_kwargs=None, finish_kwargs=None):
        states = []

        self.restart(restart_kwargs or {})
        while True:
            state = self.current_state
            action = func(state)
            r, is_over = self.action(action)
            if ret_states:
                states.append(state)
            if is_over:
                self.finish(finish_kwargs or {})
                break

        if ret_states:
            states.append(self.current_state)
            return states

    def _get_action_space(self):
        return None

    def _get_current_state(self):
        return None

    def _action(self, action):
        raise NotImplementedError()

    def _restart(self, *args, **kwargs):
        raise NotImplementedError()

    def _finish(self, *args, **kwargs):
        pass

    @property
    def unwrapped(self):
        return self


class SimpleRLEnvironBase(RLEnvironBase):
    _current_state = None

    def __init__(self):
        super().__init__()
        self._reward_history = []

    def _get_current_state(self):
        return self._current_state

    def _set_current_state(self, state):
        self._current_state = state

    def action(self, action):
        r, is_over = self._action(action)
        self._reward_history.append(r) 
        return r, is_over

    def restart(self, *args, **kwargs):
        rc = self._restart(*args, **kwargs)
        self._reward_history = []
        return rc

    def finish(self, *args, **kwargs):
        rc = self._finish(*args, **kwargs)
        self.append_stat('score', sum(self._reward_history))
        return rc


class ProxyRLEnvironBase(RLEnvironBase):
    def __init__(self, other):
        super().__init__()
        self.__proxy = other

    @property
    def proxy(self):
        return self.__proxy

    @property
    def stats(self):
        return self.__proxy.stats

    def append_stat(self, name, value):
        self.__proxy.append_stat(name)
        return self

    def clear_stats(self):
        self.__proxy.clear_stats()
        return self

    @property
    def action_space(self):
        return self.__proxy.action_space

    # directly override the action_space to disable cache
    # def _get_action_space(self):
    #     return self.__proxy.action_space

    def _get_current_state(self):
        return self.__proxy.current_state

    def _action(self, action):
        return self.__proxy.action(action)

    def _restart(self, *args, **kwargs):
        return self.__proxy.restart(*args, **kwargs)

    def _finish(self, *args, **kwargs):
        return self.__proxy.finish(*args, **kwargs)

    @property
    def unwrapped(self):
        return self.proxy.unwrapped


class ActionSpaceBase(object):
    def __init__(self, action_meanings=None):
        self.__rng = random.gen_rng()
        self._action_meanings = action_meanings

    @property
    def rng(self):
        return self.__rng

    @property
    def action_meanings(self):
        return self._action_meanings

    def sample(self, theta=None):
        return self._sample(theta)

    def _sample(self, theta=None):
        return None


class DiscreteActionSpace(ActionSpaceBase):
    def __init__(self, nr_actions, action_meanings=None):
        super().__init__(action_meanings=action_meanings)
        self._nr_actions = nr_actions

    @property
    def nr_actions(self):
        return self._nr_actions

    def _sample(self, theta=None):
        if theta is None:
            return self.rng.choice(self._nr_actions)
        return self.rng.choice(self._nr_actions, p=theta)


class ContinuousActionSpace(ActionSpaceBase):
    @staticmethod
    def __canonize_bound(v, shape):
        if type(v) is np.ndarray:
            assert v.shape == shape, 'Invalid shape for bound value: expect {}, got {}.'.format(
                    shape, v.shape)
            return v

        assert type(v) in (int, float), 'Invalid type for boudn value'
        return np.ones(shape=shape, dtype='float32') * v

    def __init__(self, low, high=None, shape=None, action_meanings=None):
        super().__init__(action_meanings=action_meanings)

        if high is None:
            low, high = -low, low

        if shape is None:
            assert low is not None and high is not None, 'Must provide low and high'
            low, high = np.array(low), np.array(high)
            assert low.shape == high.shape, 'Low and high must have smae shape, got: {} and {}'.format(
                    low.shape, high.shape)

            self._low = low
            self._high = high
            self._shape = low.shape
        else:
            self._low = self.__canonize_bound(low, shape)
            self._high = self.__canonize_bound(high, shape)
            self._shape = shape

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def shape(self):
        return self._shape

    def _sample(self, theta=None):
        if theta is not None:
            mu, std = theta
            return self.rng.randn(*self.shape) * std + mu
        return self.rng.uniform(self._low, self._high)
