# -*- coding:utf8 -*-
# File   : train.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/17/17
# 
# This file is part of TensorArtist.

from tartist.nn.train.trainer import TrainerBase
from tartist.nn import summary
from tartist.nn.train.env import TrainerEnvBase
from tartist.data.flow.base import SimpleDataFlowBase
from tartist.data.flow.collections import EmptyDictDataFlow
from tartist.core.utils.meta import notnone_property

import tensorflow as tf


class GANGraphKeys:
    GENERATOR_VARIABLES = 'generator'
    DISCRIMINATOR_VARIABLES = 'discriminator'

    GENERATOR_SUMMARIES = 'generator_summaries'
    DISCRIMINATOR_SUMMARIES = 'discriminator_summaries'


class GANTrainerEnv(TrainerEnvBase):
    _g_optimizer = None
    _d_optimizer = None

    @notnone_property
    def g_loss(self):
        return self.network.outputs['g_loss']

    @notnone_property
    def d_loss(self):
        return self.network.outputs['d_loss']

    @notnone_property
    def g_optimizer(self):
        return self._g_optimizer

    def set_g_optimizer(self, opt):
        self._g_optimizer = opt
        return self

    @notnone_property
    def d_optimizer(self):
        return self._d_optimizer

    def set_d_optimizer(self, opt):
        self._d_optimizer = opt
        return self

    def make_optimizable_func(self, d_loss=None, g_loss=None):
        # need to access collections
        with self.as_default():
            d_loss = d_loss or self.d_loss
            g_loss = g_loss or self.g_loss

            g_func = self.make_func()
            scope = GANGraphKeys.GENERATOR_VARIABLES + '/.*'
            g_var_list = self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            g_func.add_extra_op(self.g_optimizer.minimize(g_loss, var_list=g_var_list))

            d_func = self.make_func()
            scope = GANGraphKeys.DISCRIMINATOR_VARIABLES + '/.*'
            d_var_list = self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            d_func.add_extra_op(self.d_optimizer.minimize(d_loss, var_list=d_var_list))
            return g_func, d_func


class GANDataFlow(SimpleDataFlowBase):
    def __init__(self, g_input, d_input, g_times, d_times):
        super().__init__()
        self._g_input, self._d_input = g_input, d_input
        self._g_times, self._d_times = g_times, d_times

        if self._g_input is None:
            self._g_input = EmptyDictDataFlow()

    def _gen(self):
        g_it = iter(self._g_input)
        d_it = iter(self._d_input)
        while True:
            out = {'g': [], 'd': []}
            for i in range(self._g_times):
                out['g'].append(next(g_it))
            for i in range(self._d_times):
                out['d'].append(next(d_it))
            yield out


class GANTrainer(TrainerBase):
    _g_func = None
    _d_func = None

    def initialize(self):
        with self.env.as_default():
            summary.scalar('g_loss', self.env.g_loss, collections=[GANGraphKeys.GENERATOR_SUMMARIES])
            summary.scalar('d_loss', self.env.d_loss, collections=[GANGraphKeys.DISCRIMINATOR_SUMMARIES])
        self._g_func, self._d_func = self.env.make_optimizable_func()
        assert not self._g_func.queue_enabled and not self._d_func.queue_enabled
        super().initialize()

    @notnone_property
    def g_func(self):
        return self._g_func

    @notnone_property
    def d_func(self):
        return self._d_func

    def _compile_fn_train(self):
        if not self._g_func.compiled:
            summaries = self.network.get_merged_summaries(GANGraphKeys.GENERATOR_SUMMARIES)
            if summaries is not None:
                self._g_func.add_extra_kwoutput('summaries', summaries)
            self._g_func.compile({'g_loss': self.env.g_loss})

        if not self._d_func.compiled:
            summaries = self.network.get_merged_summaries(GANGraphKeys.DISCRIMINATOR_SUMMARIES)
            if summaries is not None:
                self._d_func.add_extra_kwoutput('summaries', summaries)
            self._d_func.compile({'d_loss': self.env.d_loss})

    def _run_step(self, data):
        self._compile_fn_train()
        all_summaries = []
        d_losses = []
        g_losses = []

        def process_summaries(out):
            if 'summaries' in out:
                summaries = tf.Summary.FromString(out['summaries'])
                all_summaries.append(summaries)

        for feed_dict in data['d']:
            out = self._d_func.call_args(feed_dict)
            process_summaries(out)
            d_losses.append(out['d_loss'])

        for feed_dict in data['g']:
            out = self._g_func.call_args(feed_dict)
            process_summaries(out)
            g_losses.append(out['g_loss'])

        self.runtime['summaries'] = all_summaries
        return {'g_loss': sum(g_losses) / len(g_losses), 'd_loss': sum(d_losses) / len(d_losses)}
