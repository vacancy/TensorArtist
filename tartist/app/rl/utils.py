# -*- coding:utf8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/18/17
# 
# This file is part of TensorArtist.

from .base import ProxyRLEnvironBase
from tartist.core import get_logger 
import functools
import collections
import numpy as np

logger = get_logger(__file__)


__all__ = [
        'TransparentAttributeProxyRLEnviron',
        'AutoRestartProxyRLEnviron', 
        'RepeatActionProxyRLEnviron', 'NOPFillProxyRLEnviron',
        'LimitLengthProxyRLEnviron', 
        'MapStateProxyRLEnviron', 'HistoryFrameProxyRLEnviron',
        'ManipulateRewardProxyRLEnviron', 'manipulate_reward', 
        'remove_proxies']


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


class HistoryFrameProxyRLEnviron(ProxyRLEnvironBase):
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
