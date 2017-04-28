# -*- coding:utf8 -*-
# File   : wrapper.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/28/17
# 
# This file is part of TensorArtist

from .base import get_optimizer_variable
from ..graph.env import get_default_env
from ...core import get_logger
from ...core.utils.meta import notnone_property

logger = get_logger(__file__)


class OptimizerWrapper(object):
    learning_rate_variable_name = 'learning_rate'

    def __init__(self, base_optimizer=None):
        self._base_optimizer = base_optimizer
        self._grad_modifiers = []
        self._owner_env = get_default_env()

    @notnone_property
    def base_optimizer(self):
        return self._base_optimizer

    def set_base_optimizer(self, optimizer):
        self._base_optimizer = optimizer

    @property
    def learning_rate(self):
        with self._owner_env.as_default():
            return get_optimizer_variable(self.learning_rate_variable_name, env=self._owner_env).get_value()

    def set_learning_rate(self, value):
        with self._owner_env.as_default():
            logger.critical('Setting learning rate to {} (var. name={})'.format(
                value, self.learning_rate_variable_name))
            get_optimizer_variable(self.learning_rate_variable_name, env=self._owner_env).set_value(value)

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

    def minimize(self, loss, var_list=None):
        with self._owner_env.as_default():
            # MJY(20170427): Colocate gradients to ensure speed when performing multi-tower training
            all_gradients = self._base_optimizer.compute_gradients(loss, var_list=var_list, colocate_gradients_with_ops=True)
            all_gradients = self._apply_grad_modifiers(all_gradients)
            return self._base_optimizer.apply_gradients(all_gradients)

    def _apply_grad_modifiers(self, all_gradients):
        for f in self._grad_modifiers:
            all_gradients = f(all_gradients)
        return all_gradients

