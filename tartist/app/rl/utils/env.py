# -*- coding:utf8 -*-
# File   : opr.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/18/17
#
# This file is part of TensorArtist.

from tartist.app.rl.base import DiscreteActionSpace
from tartist.app.rl.base import ProxyRLEnvironBase
from tartist.core import get_logger
from tartist.core.utils.meta import run_once
import copy
import functools
import collections
import numpy as np

logger = get_logger(__file__)


__all__ = [
        'TransparentAttributeProxyRLEnviron',
        'AutoRestartProxyRLEnviron',
        'RepeatActionProxyRLEnviron', 'NOPFillProxyRLEnviron',
        'LimitLengthProxyRLEnviron', 'MapStateProxyRLEnviron',
        'MapActionProxyRLEnviron', 'HistoryFrameProxyRLEnviron',
        'ManipulateRewardProxyRLEnviron', 'manipulate_reward',
        'remove_proxies', 'find_proxy']


class TransparentAttributeProxyRLEnviron(ProxyRLEnvironBase):
    def __getattr__(self, name):
        return getattr(remove_proxies(self), name)


class AutoRestartProxyRLEnviron(ProxyRLEnvironBase):
    def _action(self, action):
        r, is_over = self.proxy.action(action)
        if is_over:
            self.finish()
            self.restart()
        return r, is_over


class RepeatActionProxyRLEnviron(ProxyRLEnvironBase):
    def __init__(self, other, repeat):
        super().__init__(other)
        self._repeat = repeat

    def _action(self, action):
        total_r = 0
        for i in range(self._repeat):
            r, is_over = self.proxy.action(action)
            total_r += r
            if is_over:
                break
        return total_r, is_over


class NOPFillProxyRLEnviron(ProxyRLEnvironBase):
    def __init__(self, other, nr_fill, nop=0):
        super().__init__(other)
        self._nr_fill = nr_fill
        self._nop = nop

    def _action(self, action):
        total_r, is_over = self.proxy.action(action)
        for i in range(self._nr_fill):
            r, is_over = self.proxy.action(self._nop)
            total_r += r
            if is_over:
                break
        return total_r, is_over


class LimitLengthProxyRLEnviron(ProxyRLEnvironBase):
    def __init__(self, other, limit):
        super().__init__(other)
        self._limit = limit
        self._cnt = 0

    @property
    def limit(self):
        return self._limit

    def set_limit(self, limit):
        self._limit = limit
        return self

    def _action(self, action):
        r, is_over = self.proxy.action(action)
        self._cnt += 1
        if self._limit is not None and self._cnt >= self._limit:
            is_over = True
        return r, is_over

    def _restart(self, *args, **kwargs):
        super()._restart(*args, **kwargs)
        self._cnt = 0


class MapStateProxyRLEnviron(ProxyRLEnvironBase):
    def __init__(self, other, func):
        super().__init__(other)
        self._func = func

    def _get_current_state(self):
        return self._func(self.proxy.current_state)


class MapActionProxyRLEnviron(ProxyRLEnvironBase):
    def __init__(self, other, mapping):
        super().__init__(other)
        assert type(mapping) in [tuple, list]
        for i in mapping:
            assert type(i) is int
        self._mapping = mapping
        action_space = other.action_space
        assert isinstance(action_space, DiscreteActionSpace)
        action_meanings = [action_space.action_meanings[i] for i in mapping]
        self._action_space = DiscreteActionSpace(len(mapping), action_meanings)

    def _get_action_space(self):
        return self._action_space

    def _action(self, action):
        assert action < len(self._mapping)
        return self.proxy.action(self._mapping[action])


HistoryFrameProxyRLEnviron_copy_warning = run_once(lambda: logger.warn('HistoryFrameProxyRLEnviron._copy' +
    HistoryFrameProxyRLEnviron._copy_history.__doc__))
class HistoryFrameProxyRLEnviron(ProxyRLEnvironBase):
    @staticmethod
    def __zeros_like(v):
        if type(v) is tuple:
            return tuple(HistoryFrameProxyRLEnviron.__zeros_like(i) for i in v)
        assert isinstance(v, np.ndarray)
        return np.zeros_like(v, dtype=v.dtype)

    @staticmethod
    def __concat(history):
        last = history[-1]
        if type(last) is tuple:
            return tuple(HistoryFrameProxyRLEnviron.__concat(i) for i in zip(*history))
        return np.concatenate(history, axis=-1)

    def __init__(self, other, history_length):
        super().__init__(other)
        self._history = collections.deque(maxlen=history_length)

    def _get_current_state(self):
        while len(self._history) != self._history.maxlen:
            assert len(self._history) > 0
            v = self._history[-1]
            self._history.appendleft(self.__zeros_like(v))
        return self.__concat(self._history)

    def _set_current_state(self, state):
        if len(self._history) == self._history.maxlen:
            self._history.popleft()
        self._history.append(state)

    # Use shallow copy
    def _copy_history(self, _called_directly=True):
        """DEPRECATED: (2017-12-23) Use copy_history directly."""
        if _called_directly:
            HistoryFrameProxyRLEnviron_copy_warning()
        return copy.copy(self._history)

    def _restore_history(self, history, _called_directly=True):
        """DEPRECATED: (2017-12-23) Use restore_history directly."""
        if _called_directly:
            HistoryFrameProxyRLEnviron_copy_warning()
        assert isinstance(history, collections.deque)
        assert history.maxlen == self._history.maxlen
        self._history = copy.copy(history)

    def copy_history(self):
        return self._copy_history(_called_directly=False)

    def restore_history(self, history):
        return self._restore_history(history, _called_directly=False)

    def _action(self, action):
        r, is_over = self.proxy.action(action)
        self._set_current_state(self.proxy.current_state)
        return r, is_over

    def _restart(self, *args, **kwargs):
        super()._restart(*args, **kwargs)
        self._history.clear()
        self._set_current_state(self.proxy.current_state)


class ManipulateRewardProxyRLEnviron(ProxyRLEnvironBase):
    """DEPRECATED: (2017-11-20) Use manipulate_reward instead."""

    def __init__(self, other, func):
        logger.warn('ManipulateRewardProxyRLEnviron may cause wrong reward history; use manipulate_reward instead.')
        super().__init__(other)
        self._func = func

    def _action(self, action):
        r, is_over = self.proxy.action(action)
        return self._func(r), is_over


def manipulate_reward(player, func):
    old_func = player._action

    @functools.wraps(old_func)
    def new_func(action):
        r, is_over = old_func(action)
        return func(r), is_over

    player._action = new_func
    return player


def remove_proxies(environ):
    """Remove all wrapped proxy environs"""
    while isinstance(environ, ProxyRLEnvironBase):
        environ = environ.proxy
    return environ


def find_proxy(environ, proxy_cls):
    while not isinstance(environ, proxy_cls) and isinstance(environ, ProxyRLEnvironBase):
        environ = environ.proxy
    if isinstance(environ, proxy_cls):
        return environ
    return None
