# -*- coding:utf8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/18/17
# 
# This file is part of TensorArtist

from .base import ProxyRLEnvironBase


__all__ = ['AutoRestartProxyRLEnviron', 'LimitLengthProxyRLEnviron', 'MapStateProxyRLEnviron']


class AutoRestartProxyRLEnviron(ProxyRLEnvironBase):
    def _action(self, action):
        r, is_over = self.proxy.action(action)
        if is_over:
            self.finish()
            self.restart()
        return r, is_over


class LimitLengthProxyRLEnviron(ProxyRLEnvironBase):
    """limit length + auto restart"""
    def __init__(self, other, limit):
        super().__init__(other)
        self._limit = limit
        self._cnt = 0

    def _action(self, action):
        r, is_over = self.proxy.action(action)
        self._cnt += 1
        if self._cnt >= self._limit:
            is_over = True
        if is_over:
            self.finish()
            self.restart()
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
