# -*- coding:utf8 -*-
# File   : base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/18/17
# 
# This file is part of TensorArtist

from .. import random
import collections

__all__ = [
    'RLEnvironBase', 'SimpleRLEnvironBase', 'ProxyRLEnvironBase',
    'ActionSpaceBase', 'DiscreteActionSpace'
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

    @property
    def action_space(self):
        return self._get_action_space()

    @property
    def current_state(self):
        return self._get_current_state()

    def action(self, action):
        return self._action(action)

    def restart(self):
        return self._restart()

    def finish(self):
        return self._finish()

    def play_one_episode(self, func):
        self.restart()
        cnt = 0
        while True:
            cnt += 1
            state = self.current_state
            action = func(state)
            r, is_over = self.action(action)
            if cnt % 10 == 0:
                import os
                print(os.getpid(), cnt)
            if is_over:
                return self.finish()

    def _get_action_space(self):
        return None

    def _get_current_state(self):
        return None

    def _action(self, action):
        raise NotImplementedError()

    def _restart(self):
        raise NotImplementedError()

    def _finish(self):
        pass


class SimpleRLEnvironBase(RLEnvironBase):
    _current_state = None

    def _get_current_state(self):
        return self._current_state

    def _set_current_state(self, state):
        self._current_state = state


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

    def _get_action_space(self):
        return self.__proxy.action_space

    def _get_current_state(self):
        return self.__proxy.current_state

    def _action(self, action):
        return self.__proxy.action(action)

    def _restart(self):
        return self.__proxy.restart()

    def _finish(self):
        return self.__proxy.finish()


class ActionSpaceBase(object):
    def __init__(self):
        self.__rng = random.gen_rng()

    @property
    def rng(self):
        return self.__rng

    def sample(self, theta=None):
        return self._sample(theta)

    def _sample(self, theta=None):
        return None


class DiscreteActionSpace(ActionSpaceBase):
    def __init__(self, nr_actions):
        super().__init__()
        self._nr_actions = nr_actions

    @property
    def nr_actions(self):
        return self._nr_actions

    def _sample(self, theta=None):
        if theta is None:
            return self.rng.randint(self._nr_actions)
        return self.rng.choice(self._nr_actions, p=theta)

