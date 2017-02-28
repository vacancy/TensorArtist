# -*- coding:utf8 -*-
# File   : wrapper.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/28/17
# 
# This file is part of TensorArtist

from ...core.utils.meta import notnone_property


class OptimizerWrapper(object):
    def __init__(self, base_optimizer=None):
        self._base_optimizer = base_optimizer
        self._grad_modifiers = []

    @notnone_property
    def base_optimizer(self):
        return self._base_optimizer

    def set_base_optimizer(self, optimizer):
        self._base_optimizer = optimizer

    @property
    def grad_modifiers(self):
        return self._grad_modifiers

    def insert_grad_modifier(self, index, grad_modifier):
        self._grad_modifiers.insert(index, grad_modifier)
        return index

    def append_grad_modifier(self, grad_modifier):
        self._grad_modifiers.append(grad_modifier)
        return len(self._grad_modifiers) - 1

    def pop_grad_modifier(self, index=None):
        return self._grad_modifiers.pop(index)

    def minimize(self, loss):
        all_gradients = self._base_optimizer.compute_gradients(loss)
        all_gradients = self._apply_grad_modifiers(all_gradients)
        return self._base_optimizer.apply_gradients(all_gradients)

    def _apply_grad_modifiers(self, all_gradients):
        for f in self._grad_modifiers:
            all_gradients = f(all_gradients)
        return all_gradients

