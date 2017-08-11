# -*- coding:utf8 -*-
# File   : evolution.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/08/2017
# 
# This file is part of TensorArtist.

from tartist import random
from tartist.core.utils.meta import notnone_property
from tartist.nn import opr as O
from tartist.nn.tfutils import escape_name
from tartist.nn.graph import as_varnode, as_tftensor
from tartist.nn.optimizer import CustomOptimizerBase
from tartist.nn.train import TrainerBase

import tensorflow as tf
import numpy as np

__all__ = ['EvolutionBasedOptimizerBase', 'CEMOptimizer', 'ESOptimizer', "EvolutionBasedTrainer"]


class EvolutionBasedOptimizerBase(CustomOptimizerBase):
    _name = 'evolution_optimizer'
    _var_list = None

    _param_nr_elems = None
    _param_getter = None
    _param_setter = None
    _param_provider = None

    _populations = None

    def __init__(self, env):
        self._env = env
        self.__rng = random.gen_rng()
        self.__initialized = False

    @property
    def env(self):
        return self._env

    @property
    def rng(self):
        return self.__rng

    def minimize(self):
        """Analog to preparation call in typical gradient based optimizers."""
        assert not self.__initialized, 'EvolutionOptimizer `{}` can only be used once.'.format(self._name)

        self._var_list = self._env.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self._initialize_ops(self._var_list)
        self._initialize_param()

    def register_snapshot_parts(self, env):
        env.add_snapshot_part(self._name, self._dump_param, self._load_param)

    @notnone_property
    def param_nr_elems(self):
        return self._param_nr_elems

    def get_flat_param(self):
        return self._env.session.run([self._param_getter])[0]

    def set_flat_param(self, params):
        return self._env.session.run([self._param_setter], feed_dict={
            self._param_provider: params
        })

    def _initialize_ops(self, var_list):
        var_shapes = [as_tftensor(v).get_shape().as_list() for v in var_list]
        for vs, v in zip(var_shapes, var_list):
            assert None not in vs, 'Could not determine the shape for optimizable variable: {}.'.format(v)
        var_nr_elems = [as_tftensor(v).get_shape().num_elements() for v in var_list]
        nr_total_elems = sum(var_nr_elems)

        self._param_nr_elems = nr_total_elems

        with self._env.name_scope(self._name):
            # Parameter getter
            flat_variables = [as_varnode(v).flatten(name='flat_{}'.format(escape_name(v))) for v in var_list]
            self._param_getter = as_tftensor(O.concat(flat_variables, axis=0))

            # Parameter setter
            flat_variables_tensor = O.placeholder('flat_variable_tensor', shape=(nr_total_elems, ))
            variable_assigns = []

            index = 0
            for v, vs, vn in zip(var_list, var_shapes, var_nr_elems):
                value = flat_variables_tensor[index:index+vn].reshape(vs)
                # Use tf.assign because tf.group use non-3rdparty-compatible codes.
                variable_assigns.append(tf.assign(v, value, name='assign_{}'.format(escape_name(v))))
                index += vn

            self._param_setter = tf.group(*variable_assigns)
            self._param_provider = as_tftensor(flat_variables_tensor)

    def _initialize_param(self):
        raise NotImplementedError()

    def _dump_param(self):
        raise NotImplementedError()

    def _load_param(self, param):
        raise NotImplementedError()

    def sample_flat_param(self, i, n):
        raise NotImplementedError()

    def before_epoch(self):
        self._populations = []

    def on_epoch_data(self, param, score):
        i = len(self._populations)
        self._populations.append((score, i, param))

    def after_epoch(self):
        raise NotImplementedError()


class CEMOptimizer(EvolutionBasedOptimizerBase):
    _name = 'cem_optimizer'

    # Model parameters: public access
    param_mean = None
    param_std = None

    def __init__(self, env, top_frac, initial_std=0.1):
        super().__init__(env)
        self._top_frac = top_frac
        self._initial_std = initial_std

    def _initialize_param(self):
        self.param_mean = np.zeros(shape=(self.param_nr_elems, ), dtype='float32')
        self.param_std = np.ones(shape=(self.param_nr_elems, ), dtype='float32')
        self.param_std *= self._initial_std

    def _dump_param(self):
        return self.param_mean, self.param_std

    def _load_param(self, p):
        assert self._param_nr_elems == p[0].shape[0] == p[1].shape[0]
        self.param_mean, self.param_std = p

    def sample_flat_param(self, i, n):
        return self.rng.normal(self.param_mean, self.param_std, size=(self.param_nr_elems, ))

    def after_epoch(self):
        top_n = int(len(self._populations) * self._top_frac)
        self._populations.sort(reverse=True)
        active_populations = [p[2] for p in self._populations[:top_n]]
        p1m = sum(active_populations) / top_n
        p2m = sum(map(lambda x: x ** 2, active_populations)) / top_n

        self.param_mean = p1m
        self.param_std = np.sqrt(p2m - p1m ** 2)

        self.set_flat_param(self.param_mean)


class ESOptimizer(EvolutionBasedOptimizerBase):
    _name = 'es_optimizer'

    # Model parameters: public access
    param_mean = None

    def __init__(self, env, learning_rate, noise_std=0.1):
        super().__init__(env)
        self._learning_rate = learning_rate
        self._noise_std = noise_std

    @property
    def learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, lr):
        self._learning_rate = lr

    @property
    def noise_std(self):
        return self._noise_std

    def set_noise_std(self, std):
        self._noise_std = std

    def _initialize_param(self):
        self.param_mean = np.zeros(shape=(self.param_nr_elems, ), dtype='float32')

    def _dump_param(self):
        return self.param_mean

    def _load_param(self, p):
        assert self._param_nr_elems == p.shape[0]
        self.param_mean = p

    def sample_flat_param(self, i, n):
        return self.rng.normal(self.param_mean, self.noise_std, size=(self.param_nr_elems, ))

    def after_epoch(self):
        scores = np.array([p[0] for p in self._populations], dtype='float32')
        advantage = (scores - np.mean(scores)) / np.std(scores)

        gradient = np.zeros_like(self.param_mean)
        for _, i, p in self._populations:
            gradient += (p - self.param_mean) * advantage[i]

        self.param_mean += self._learning_rate / (len(scores) * self._noise_std) * gradient


class EvolutionBasedTrainer(TrainerBase):
    _pred_func = None
    _evaluator = None

    def initialize(self):
        """Actual initialization before the optimization steps."""
        assert isinstance(self.optimizer, EvolutionBasedOptimizerBase)

        super().initialize()
        self._initialize_pred_func()
        with self.env.as_default():
            self.optimizer.minimize()
        self.optimizer.register_snapshot_parts(self.env)

    def _initialize_pred_func(self):
        """Compile the function."""
        self._pred_func = self.env.make_func()
        self._pred_func.compile(self.env.network.outputs)

    @notnone_property
    def pred_func(self):
        return self._pred_func

    @notnone_property
    def evaluator(self):
        return self._evaluator

    def set_evaluator(self, evaluator):
        self._evaluator = evaluator

    def _wrapped_run_step(self):
        if self.runtime['iter'] % self.epoch_size == 1:
            self.trigger_event('epoch:before')
            self.optimizer.before_epoch()

        self.trigger_event('iter:before', {})
        out = self._run_step(None)
        self.trigger_event('iter:after', {}, out)

        if self.runtime['iter'] % self.epoch_size == 0:
            self.optimizer.after_epoch()
            self.trigger_event('epoch:after')

    def _run_step(self, _):
        param = self.optimizer.sample_flat_param(self.iter_in_epoch, self.epoch_size)
        self.optimizer.set_flat_param(param)
        score = self.evaluator(self)
        self.optimizer.on_epoch_data(param, score)

        summaries = tf.Summary()
        summaries.value.add(tag='train/score', simple_value=score)
        self.runtime['summaries'] = summaries
        return dict(score=score)
