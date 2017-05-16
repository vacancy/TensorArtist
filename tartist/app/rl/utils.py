# -*- coding:utf8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/18/17
# 
# This file is part of TensorArtist

from .base import ProxyRLEnvironBase


__all__ = ['AutoRestartProxyRLEnviron', 
        'RepeatActionProxyRLEnviron', 'NOPFillProxyRLEnviron',
        'LimitLengthProxyRLEnviron', 'MapStateProxyRLEnviron', 
        'remove_proxies']


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
        if self._cnt >= self._limit:
            is_over = True
        return r, is_over

    def _restart(self):
        super()._restart()
        self._cnt = 0


class MapStateProxyRLEnviron(ProxyRLEnvironBase):
    def __init__(self, other, func):
        super().__init__(other)
        self._func = func

    def _get_current_state(self):
        return self._func(self.proxy.current_state)


def remove_proxies(environ):
    """Remove all wrapped proxy environs"""
    while isinstance(environ, ProxyRLEnvironBase):
        environ = environ.proxy
    return environ

