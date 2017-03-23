# -*- coding:utf8 -*-
# File   : gym.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/18/17
# 
# This file is part of TensorArtist

from .base import SimpleRLEnvironBase, DiscreteActionSpace, ProxyRLEnvironBase
import threading
import numpy as np
import collections
import os
import errno

try:
    import gym
    import gym.wrappers
except ImportError:
    gym = None

_ENV_LOCK = threading.Lock()


def get_env_lock():
    return _ENV_LOCK

__all__ = ['GymRLEnviron', 'GymHistoryProxyRLEnviron', 'GymPreventStuckProxyRLEnviron']


class GymRLEnviron(SimpleRLEnvironBase):
    def __init__(self, name, vis=False, dump_dir=None):
        super().__init__()

        with get_env_lock():
            self._gym = gym.make(name)

        if dump_dir:
            if dump_dir == '' or os.path.isdir(dump_dir):
                continue
            try:
                os.makedirs(dump_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise e

            self._gym = gym.wrappers.Monitor(self._gym, dumpdir)

        self._vis = vis
        self._reward_history = []

    def _get_action_space(self):
        spc = self._gym.action_space
        assert isinstance(spc, gym.spaces.discrete.Discrete)
        return DiscreteActionSpace(spc.n)

    def _action(self, action):
        o, r, is_over, info = self._gym.step(action)
        self._reward_history.append(r)
        self._set_current_state(o)
        if self._vis:
            self._gym.render()
            time.sleep(0.1)
        return r, is_over

    def _restart(self):
        o = self._gym.reset()
        self._set_current_state(o)
        self._reward_history = []

    def _finish(self):
        self.append_stat('score', sum(self._reward_history))


class GymHistoryProxyRLEnviron(ProxyRLEnvironBase):
    def __init__(self, other, history_length):
        super().__init__(other)
        self._history = collections.deque(maxlen=history_length)

    def _get_current_state(self):
        while len(self._history) != self._history.maxlen:
            assert len(self._history) > 0
            v = self._history[-1]
            self._history.appendleft(np.zeros_like(v, dtype=v.dtype))
        return np.concatenate(self._history, axis=2)

    def _set_current_state(self, state):
        if len(self._history) == self._history.maxlen:
            self._history.popleft()
        self._history.append(state)

    def _action(self, action):
        r, is_over = self.proxy.action(action)
        self._set_current_state(self.proxy.current_state)
        return r, is_over

    def _restart(self):
        self.proxy.restart()
        self._history.clear()
        self._set_current_state(self.proxy.current_state)


class GymPreventStuckProxyRLEnviron(ProxyRLEnvironBase):
    def __init__(self, other, max_repeat, action):
        super().__init__(other)
        self._action_list = collections.deque(maxlen=max_repeat)
        self._insert_action = action

    def _action(self, action):
        self._action_list.append(action)
        if self._action_list.count(self._action_list[0]) == self._action_list.maxlen:
            action = self._insert_action
        r, is_over = self.proxy.action(action)
        if is_over:
            self._action_list.clear()
        return r, is_over

    def _restart(self):
        self.proxy.restart()
        self._action_list.clear()

