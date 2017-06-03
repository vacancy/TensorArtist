# -*- coding:utf8 -*-
# File   : gym.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/18/17
# 
# This file is part of TensorArtist.

from .base import SimpleRLEnvironBase, ProxyRLEnvironBase
from .base import DiscreteActionSpace, ContinuousActionSpace
from tartist.core import io
import copy
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
    def __init__(self, name, dump_dir=None, force_dump=False, state_mode='DEFAULT'):
        super().__init__()

        with get_env_lock():
            self._gym = gym.make(name)

        if dump_dir:
            io.mkdir(dump_dir)
            self._gym = gym.wrappers.Monitor(self._gym, dump_dir, force=force_dump)

        assert state_mode in ('DEFAULT', 'RENDER', 'BOTH')
        self._state_mode = state_mode

    @property
    def gym(self):
        return self._gym

    def render(self, mode='human', close=False):
        return self._gym.render(mode=mode, close=close)

    def _set_current_state(self, o):
        if self._state_mode == 'DEFAULT':
            pass
        else:
            rendered = self.render('rgb_array')
            if self._state_mode == 'RENDER':
                o = rendered
            else:
                o = (o, rendered)
        super()._set_current_state(o)

    def _get_action_space(self):
        spc = self._gym.action_space

        if isinstance(spc, gym.spaces.discrete.Discrete):
            try:
                action_meanings = self._gym.get_action_meanings()
            except AttributeError:
                if 'Atari' in self._gym.unwrapped.__class__.__name__:
                    from gym.envs.atari.atari_env import ACTION_MEANING
                    action_meanings = [ACTION_MEANING[i] for i in range(spc.n)]
                else:
                    action_meanings = ['unknown{}'.format(i) for i in range(spc.n)]
            return DiscreteActionSpace(spc.n, action_meanings=action_meanings)
        elif isinstance(spc, gym.spaces.box.Box):
            return ContinuousActionSpace(spc.low, spc.high, spc.shape)
        else:
            raise ValueError('Unknown gym space spec: {}.'.format(spc))

    def _action(self, action):
        # hack for continuous control
        if type(action) in (tuple, list):
            action = np.array(action)

        o, r, is_over, info = self._gym.step(action)
        self._set_current_state(o)
        return r, is_over

    def _restart(self):
        o = self._gym.reset()
        self._set_current_state(o)

    def _finish(self):
        self._gym.close()


class GymHistoryProxyRLEnviron(ProxyRLEnvironBase):
    def __init__(self, other, history_length):
        super().__init__(other)
        self._history = collections.deque(maxlen=history_length)

    def _get_current_state(self):
        while len(self._history) != self._history.maxlen:
            assert len(self._history) > 0
            v = self._history[-1]
            self._history.appendleft(np.zeros_like(v, dtype=v.dtype))
        return np.concatenate(self._history, axis=-1)

    def _set_current_state(self, state):
        if len(self._history) == self._history.maxlen:
            self._history.popleft()
        self._history.append(state)

    def _copy_history(self):
        return copy.copy(self._history)

    def _restore_history(self, history):
        assert isinstance(history, collections.deque)
        assert history.maxlen == self._history.maxlen
        self._history = copy.copy(history)

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
