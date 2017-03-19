# -*- coding:utf8 -*-
# File   : gym.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/18/17
# 
# This file is part of TensorArtist

from .base import SimpleRLEnvironBase, DiscreteActionSpace
import threading

try:
    import gym
    import gym.wrappers
except ImportError:
    gym = None

_ENV_LOCK = threading.Lock()


def get_env_lock():
    return _ENV_LOCK


class GymRLEnviron(SimpleRLEnvironBase):
    def __init__(self, name, dump_dir=None):
        super().__init__()

        with get_env_lock():
            self._gym = gym.make(name)

        # TODO(MJY): support monitor
        assert dump_dir is None

        self._reward_history = []

    def _get_action_space(self):
        spc = self._gym.actoin_space
        assert isinstance(spc, gym.spaces.discrete.Discrete)
        return DiscreteActionSpace(spc.n)

    def _action(self, action):
        o, r, is_over, info = self._gym.step(action)
        self._reward_history.append(r)
        self._set_current_state(o)
        return r, is_over

    def restart_episode(self):
        o = self._gym.reset()
        self._set_current_state(o)
        self._reward_history = []

    def finish_episode(self):
        self.append_stat('score', sum(self._reward_history))
