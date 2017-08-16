# -*- coding:utf8 -*-
# File   : trpo.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/08/2017
# 
# This file is part of TensorArtist.

from .utils import vectorize_var_list
from ..math_utils import normalize_advantage
from tartist.core.utils.meta import notnone_property
from tartist.data.flow import SimpleDataFlowBase
from tartist.nn import summary
from tartist.nn.graph import as_tftensor
from tartist.nn.tfutils import escape_name
from tartist.nn.optimizer import CustomOptimizerBase
from tartist.nn.train import TrainerEnvBase, TrainerBase

import numpy as np
import tensorflow as tf

__all__ = ['TRPOGraphKeys', 'TRPODataFlow', 'TRPOOptimizer', 'TRPOTrainerEnv', 'TRPOTrainer']


class TRPOGraphKeys:
    POLICY_VARIABLES = 'policy'
    VALUE_VARIABLES = 'value'

    POLICY_SUMMARIES = 'policy_summaries'
    VALUE_SUMMARIES = 'value_summaries'


class TRPODataFlow(SimpleDataFlowBase):
    def __init__(self, collector, target, incl_value):
        self._collector = collector
        self._target = target
        self._incl_value = incl_value

        assert self._collector.mode == 'EPISODE'

    def _initialize(self):
        self._collector.initialize()

    def _gen(self):
        while True:
            data = self._collector.collect(self._target)
            data = self._process(data)
            yield data

    def _process(self, raw_data):
        data_list = []
        for t in raw_data:
            data = dict(
                step=[],
                state=[],
                action=[],
                theta_old=[],
                reward=[],
                value=[],
                score=0
            )

            for i, e in enumerate(t):
                data['step'].append(i)
                data['state'].append(e.state)
                data['action'].append(e.action)
                data['theta_old'].append(e.outputs['theta'])
                data['reward'].append(e.reward)
                data['score'] += e.reward

                if self._incl_value:
                    data['value'].append(e.outputs['value'])

            if not self._incl_value:
                del data['value']

            for k, v in data.items():
                data[k] = np.array(v)

            if len(t) > 0:
                data_list.append(data)

        return data_list


class TRPOOptimizer(CustomOptimizerBase):
    _name = 'trpo_optimizer'

    _cg_max_nr_iters = 10
    _cg_residual_tol = 1e-8
    _cg_eps = 1e-8

    def __init__(self, env, max_kl, cg_damping):
        self._env = env
        self._max_kl = max_kl
        self._cg_damping = cg_damping

    def minimize(self, loss, kl_self, var_list):
        with self._env.name_scope(self._name):
            pg_grad = vectorize_var_list(tf.gradients(-loss, var_list))
            kl_grad = vectorize_var_list(tf.gradients(kl_self, var_list))

            with tf.name_scope('conjugate_gradient'):
                full_step, _ = self._conjugate_gradient(var_list, pg_grad, kl_grad, self._max_kl, self._cg_damping)

            with tf.name_scope('update'):
                opt_op = self._gen_update_op(var_list, full_step)

        # TODO:: Line search
        return opt_op

    def _gen_update_op(self, var_list, full_step):
        var_shapes = [as_tftensor(v).get_shape().as_list() for v in var_list]
        for vs, v in zip(var_shapes, var_list):
            assert None not in vs, 'Could not determine the shape for optimizable variable: {}.'.format(v)
        var_nr_elems = [as_tftensor(v).get_shape().num_elements() for v in var_list]
        var_assigns = []

        index = 0
        for v, vs, vn in zip(var_list, var_shapes, var_nr_elems):
            value = tf.reshape(full_step[index:index+vn], vs)
            # Use tf.assign because tf.group use non-3rdparty-compatible codes.
            var_assigns.append(tf.assign_add(v, value, name='assign_add_{}'.format(escape_name(v))))
            index += vn

        return tf.group(*var_assigns)

    def _conjugate_gradient(self, var_list, pg_grads, kl_grads, max_kl, cg_damping):
        """Construct conjugate gradient descent algorithm inside computation graph for improved efficiency.
        From: https://github.com/steveKapturowski/tensorflow-rl/blob/master/algorithms/trpo_actor_learner.py"""
        i0 = tf.constant(0, dtype='int32')

        def loop_condition(i, r, p, x, rdotr):
            return tf.logical_and(tf.greater(rdotr, self._cg_residual_tol), tf.less(i, self._cg_max_nr_iters))

        def loop_body(i, r, p, x, rdotr):
            fvp = vectorize_var_list(tf.gradients(
                tf.reduce_sum(tf.stop_gradient(p) * kl_grads),
                var_list
            ))

            z = fvp + cg_damping * p
            alpha = rdotr / (tf.reduce_sum(p * z) + self._cg_eps)
            x += alpha * p
            r -= alpha * z

            new_rdotr = tf.reduce_sum(r * r)
            beta = new_rdotr / (rdotr + self._cg_eps)
            p = r + beta * p

            # new_rdotr = tf.Print(new_rdotr, [i, new_rdotr], '[ConjugateGradient] Iteration / Residual: ')
            return i + 1, r, p, x, new_rdotr

        # Solve fvp = pg_grads
        _, r, p, stepdir, rdotr = tf.while_loop(
            loop_condition, loop_body,
            loop_vars=[i0, pg_grads, pg_grads, tf.zeros_like(pg_grads), tf.reduce_sum(pg_grads * pg_grads)])

        # Let stepdir = change in theta / direction that theta changes in
        # KL divergence approximated by 0.5 x stepdir_transpose * [Fisher Information Matrix] * stepdir
        # where the [Fisher Information Matrix] acts like a metric
        # ([Fisher Information Matrix] * stepdir) is computed using the function,
        # and then stepdir * [above] is computed manually.
        # step * Hessian * step
        fvp = vectorize_var_list(tf.gradients(
            tf.reduce_sum(tf.stop_gradient(stepdir) * kl_grads),
            var_list
        ))
        shs = 0.5 * tf.reduce_sum(stepdir * fvp)

        lm = tf.sqrt(max_kl / (shs + 1e-8))
        full_step = stepdir * lm
        negg_dot_stepdir = tf.reduce_sum(pg_grads * stepdir) / lm

        # full_step = tf.Print(full_step, [shs, max_kl, lm, full_step], '[ConjugateGradient] Result: ')

        return full_step, negg_dot_stepdir


class TRPOTrainerEnv(TrainerEnvBase):
    _policy_optimizer = None
    _value_optimizer = None

    @property
    def optimizer(self):
        return self.policy_optimizer

    @notnone_property
    def policy_optimizer(self):
        return self._policy_optimizer

    def set_policy_optimizer(self, opt):
        self._policy_optimizer = opt
        assert isinstance(self._policy_optimizer, TRPOOptimizer)
        return self

    @notnone_property
    def value_optimizer(self):
        return self._value_optimizer

    def set_value_optimizer(self, opt):
        self._value_optimizer = opt
        return self

    @notnone_property
    def policy_loss(self):
        return self.network.outputs['policy_loss']

    @notnone_property
    def policy_kl(self):
        return self.network.outputs['kl']

    @notnone_property
    def value_loss(self):
        return self.network.outputs['value_loss']

    @property
    def value_regressor(self):
        return getattr(self.network, 'value_regressor', None)

    def make_optimizable_func(self, policy_loss=None, kl_self=None, value_loss=None):
        with self.as_default():
            policy_loss = policy_loss or self.network.outputs['policy_loss']
            kl_self = kl_self or self.network.outputs['kl_self']

            p_func = self.make_func()
            scope = TRPOGraphKeys.POLICY_VARIABLES + '/.*'
            p_var_list = self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            p_func.add_extra_op(self.policy_optimizer.minimize(policy_loss, kl_self, var_list=p_var_list))

            if self.value_regressor is not None:
                return p_func

            value_loss = value_loss or self.network.outputs['value_loss']

            v_func = self.make_func()
            scope = TRPOGraphKeys.VALUE_VARIABLES + '/.*'
            v_var_list = self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            v_func.add_extra_op(self.value_optimizer.minimize(value_loss, var_list=v_var_list))
            return p_func, v_func


class TRPOTrainer(TrainerBase):
    _p_func = None
    _v_func = None
    _adv_computer = None

    def _has_value_regressor(self):
        return self._value_regressor is not None

    @property
    def _value_regressor(self):
        return self.env.value_regressor

    @notnone_property
    def adv_computer(self):
        return self._adv_computer

    def set_adv_computer(self, adv):
        self._adv_computer = adv

    def initialize(self):
        with self.env.as_default():
            summary.scalar('policy_loss', self.env.policy_loss, collections=[TRPOGraphKeys.POLICY_SUMMARIES])
            summary.scalar('policy_kl', self.env.policy_kl, collections=[TRPOGraphKeys.POLICY_SUMMARIES])
            if not self._has_value_regressor():
                summary.scalar('value_loss', self.env.value_loss, collections=[TRPOGraphKeys.VALUE_SUMMARIES])

        super().initialize()
        self._initialize_opt_func()
        self._initialize_snapshot_parts()

    def _initialize_opt_func(self):
        assert isinstance(self.env, TRPOTrainerEnv)
        if self._has_value_regressor():
            self._p_func = self.env.make_optimizable_func()
        else:
            self._p_func, self._v_func = self.env.make_optimizable_func()
        self._compile_fn_train()

    def _initialize_snapshot_parts(self):
        if self._has_value_regressor():
            self._value_regressor.register_snapshot_parts(self.env)

    def _compile_fn_train(self):
        self._compile_func_with_summary(
            self._p_func, {'p_loss': self.env.policy_loss, 'p_kl': self.env.policy_kl}, TRPOGraphKeys.POLICY_SUMMARIES)

        if self._value_regressor is None:
            self._compile_func_with_summary(
                self._v_func, {'v_loss': self.env.value_loss}, TRPOGraphKeys.VALUE_SUMMARIES)

    def _run_step(self, data_list):
        for data in data_list:
            if self._has_value_regressor():
                if 'value' not in data:
                    data['value'] = self._value_regressor.predict(data['state'], data['step'])
            else:
                assert 'value' in data

        for data in data_list:
            self.adv_computer(data)

        feed_dict = {}
        for k in ['step', 'state', 'action', 'theta_old', 'return_', 'advantage']:
            feed_dict[k] = np.concatenate([data[k] for data in data_list])

        feed_dict['advantage'] = normalize_advantage(feed_dict['advantage'])
        avg_score = sum([data['score'] for data in data_list]) / len(data_list)

        # Fit the baseline.
        if self._has_value_regressor():
            self._value_regressor.fit(feed_dict['state'], feed_dict['step'], feed_dict['return_'])
            v_outputs = None
        else:
            v_outputs = self._v_func(state=feed_dict['state'], value_label=feed_dict['return_'])

        # Policy gradient
        p_outputs = self._p_func(state=feed_dict['state'], action=feed_dict['action'],
                                 advantage=feed_dict['advantage'], theta_old=feed_dict['theta_old'])

        summaries = tf.Summary()
        summaries.value.add(tag='train/score', simple_value=avg_score)

        output = dict(score=avg_score, p_loss=p_outputs['p_loss'], p_kl=p_outputs['p_kl'])
        if v_outputs is not None:
            if 'summaries' in v_outputs:
                summaries.value.MergeFrom(tf.Summary.FromString(v_outputs['summaries']).value)
            output['v_loss'] = v_outputs['v_loss']

        if 'summaries' in p_outputs:
            summaries.value.MergeFrom(tf.Summary.FromString(p_outputs['summaries']).value)

        self.runtime['summaries'] = summaries
        return output
