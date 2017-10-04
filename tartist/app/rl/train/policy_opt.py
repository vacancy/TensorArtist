# -*- coding:utf8 -*-
# File   : policy_opt.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/08/2017
# 
# This file is part of TensorArtist.

from .opr import vectorize_var_list
from tartist import random
from tartist.app.rl.utils.math import normalize_advantage
from tartist.core.utils.meta import notnone_property
from tartist.core.utils.nd import nd_batch_size
from tartist.core.utils.thirdparty import get_tqdm_defaults
from tartist.nn import summary
from tartist.nn.graph import as_tftensor
from tartist.nn.optimizer import CustomOptimizerBase
from tartist.nn.tfutils import escape_name
from tartist.nn.train import TrainerEnvBase, TrainerBase

import numpy as np
import tensorflow as tf
from tqdm import tqdm

__all__ = [
    'ACGraphKeys',
    'TRPOOptimizer',
    'TRPOTrainerEnv', 'PPOTrainerEnv', 'PPOTrainerEnvV2',
    'TRPOTrainer', 'PPOTrainer', 'PPOTrainerV2'
]


class ACGraphKeys:
    POLICY_VARIABLES = 'policy'
    VALUE_VARIABLES = 'value'

    POLICY_SUMMARIES = 'policy_summaries'
    VALUE_SUMMARIES = 'value_summaries'


class TRPOOptimizer(CustomOptimizerBase):
    _name = 'trpo_optimizer'

    _cg_max_nr_iters = 10
    _cg_residual_tol = 1e-10
    # _cg_eps = 1e-8
    _cg_eps = 0

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
        z = fvp + cg_damping * stepdir
        shs = 0.5 * tf.reduce_sum(stepdir * z)

        lm = tf.sqrt(max_kl / (shs + 1e-8))
        full_step = stepdir * lm
        negg_dot_stepdir = tf.reduce_sum(pg_grads * stepdir) / lm

        # full_step = tf.Print(full_step, [shs, max_kl, lm, full_step], '[ConjugateGradient] Result: ')

        return full_step, negg_dot_stepdir


class ACOptimizationTrainerEnvBase(TrainerEnvBase):
    @notnone_property
    def policy_loss(self):
        return self.network.outputs['policy_loss']

    @notnone_property
    def value_loss(self):
        return self.network.outputs['value_loss']

    @property
    def value_regressor(self):
        return getattr(self.network, 'value_regressor', None)


class AlterACOptimizationTrainerEnvBase(ACOptimizationTrainerEnvBase):
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
        return self

    @notnone_property
    def value_optimizer(self):
        return self._value_optimizer

    def set_value_optimizer(self, opt):
        self._value_optimizer = opt
        return self


class JointACOptimizationTrainerEnvBase(ACOptimizationTrainerEnvBase):
    _optimizer = None

    @notnone_property
    def optimizer(self):
        return self._optimizer

    def set_optimizer(self, opt):
        self._optimizer = opt
        return self

    def make_optimizable_func(self, loss=None):
        loss = loss or self.network.loss
        loss = as_tftensor(loss)

        func = self.make_func()
        func.add_extra_op(self.optimizer.minimize(loss))
        return func


class TRPOTrainerEnv(AlterACOptimizationTrainerEnvBase):
    @notnone_property
    def policy_kl(self):
        return self.network.outputs['kl']

    @notnone_property
    def policy_kl_self(self):
        return self.network.outputs['kl_self']

    def make_optimizable_func(self, policy_loss=None, kl_self=None, value_loss=None):
        assert isinstance(self._policy_optimizer, TRPOOptimizer)
        with self.as_default():
            policy_loss = policy_loss or self.policy_loss
            kl_self = kl_self or self.policy_kl_self

            p_func = self.make_func()
            scope = ACGraphKeys.POLICY_VARIABLES + '/.*'
            p_var_list = self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            p_func.add_extra_op(self.policy_optimizer.minimize(policy_loss, kl_self, var_list=p_var_list))

            if self.value_regressor is not None:
                return p_func

            value_loss = value_loss or self.value_loss

            v_func = self.make_func()
            scope = ACGraphKeys.VALUE_VARIABLES + '/.*'
            v_var_list = self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            v_func.add_extra_op(self.value_optimizer.minimize(value_loss, var_list=v_var_list))
            return p_func, v_func


class PPOTrainerEnv(AlterACOptimizationTrainerEnvBase):
    def make_optimizable_func(self, policy_loss=None, value_loss=None):
        with self.as_default():
            policy_loss = policy_loss or self.policy_loss
            p_func = self.make_func()
            scope = ACGraphKeys.POLICY_VARIABLES + '/.*'
            p_var_list = self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            p_func.add_extra_op(self.policy_optimizer.minimize(policy_loss, var_list=p_var_list))

            if self.value_regressor is not None:
                return p_func

            value_loss = value_loss or self.network.outputs['value_loss']
            v_func = self.make_func()
            scope = ACGraphKeys.VALUE_VARIABLES + '/.*'
            v_var_list = self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            v_func.add_extra_op(self.value_optimizer.minimize(value_loss, var_list=v_var_list))
            return p_func, v_func


class PPOTrainerEnvV2(JointACOptimizationTrainerEnvBase):
    pass


class ACOptimizationTrainerBase(TrainerBase):
    _adv_computer = None

    @notnone_property
    def adv_computer(self):
        return self._adv_computer

    def set_adv_computer(self, adv):
        self._adv_computer = adv

    def _has_value_regressor(self):
        return self._value_regressor is not None

    @property
    def _value_regressor(self):
        return self.env.value_regressor

    def _gen_advantage(self, data_list):
        # Compute the value estimation and advantage.
        for data in data_list:
            if self._has_value_regressor():
                if 'value' not in data:
                    data['value'] = self._value_regressor.predict(data['state'], data['step'])
            else:
                assert 'value' in data

        for data in data_list:
            self.adv_computer(data)

    def initialize(self):
        self._initialize_summaries()
        self._initialize_opt_func()
        self._initialize_snapshot_parts()
        super().initialize()

        self._compile_fn_train()

    def _initialize_summaries(self):
        raise NotImplementedError()

    def _initialize_opt_func(self):
        raise NotImplementedError()

    def _initialize_snapshot_parts(self):
        if self._has_value_regressor():
            self._value_regressor.register_snapshot_parts(self.env)

    def _compile_fn_train(self):
        raise NotImplementedError()

    def _run_step(self, data):
        raise NotImplementedError()

    def _run_step_v_regressor(self, feed_dict):
        self._value_regressor.fit(feed_dict['state'], feed_dict['step'], feed_dict['return_'])
        return None


class SurrOptimizerTrainerFeederMixin(object):
    _feed_dict_keys = ['step', 'state', 'action', 'theta_old', 'return_', 'advantage']

    def _get_feed_dict(self, data_list):
        feed_dict = {}
        for k in self._feed_dict_keys:
            feed_dict[k] = np.concatenate([data[k] for data in data_list])
        feed_dict['advantage'] = normalize_advantage(feed_dict['advantage'])
        return feed_dict


class AlterSurrOptimizationTrainerBase(ACOptimizationTrainerBase, SurrOptimizerTrainerFeederMixin):
    _p_func = None
    _p_func_inference = None
    _v_func = None

    def _initialize_opt_func(self):
        assert isinstance(self.env, AlterACOptimizationTrainerEnvBase)
        if self._has_value_regressor():
            self._p_func = self.env.make_optimizable_func()
        else:
            self._p_func, self._v_func = self.env.make_optimizable_func()
        self._p_func_inference = self.env.make_func()

    def _run_step(self, data_list):
        self._gen_advantage(data_list)
        feed_dict = self._get_feed_dict(data_list)
        v_outputs = self._run_step_v(feed_dict)
        p_outputs = self._run_step_p(feed_dict)

        # Summaries
        avg_score = sum([data['score'] for data in data_list]) / len(data_list)
        output = dict(score=avg_score)
        summaries = tf.Summary()
        summaries.value.add(tag='train/score', simple_value=avg_score)

        if v_outputs is not None:
            if 'summaries' in v_outputs:
                summaries.value.MergeFrom(tf.Summary.FromString(v_outputs['summaries']).value)
            output.update(v_outputs)

        if 'summaries' in p_outputs:
            summaries.value.MergeFrom(tf.Summary.FromString(p_outputs['summaries']).value)
            output.update(p_outputs)

        self.runtime['summaries'] = summaries
        return output

    def _run_step_v(self, feed_dict):
        # Fit the baseline.
        if self._has_value_regressor():
            return self._run_step_v_regressor(feed_dict)
        else:
            return self._run_step_v_network(feed_dict)

    def _run_step_v_network(self, feed_dict):
        v_outputs = self._v_func(state=feed_dict['state'], value_label=feed_dict['return_'])
        return v_outputs

    def _initialize_summaries(self):
        raise NotImplementedError()

    def _compile_fn_train(self):
        raise NotImplementedError()

    def _run_step_p(self, feed_dict):
        raise NotImplementedError()


class JointSurrOptimizationTrainerBase(ACOptimizationTrainerBase, SurrOptimizerTrainerFeederMixin):
    _opt_func = None
    _inference_func = None

    def _initialize_opt_func(self):
        assert isinstance(self.env, JointACOptimizationTrainerEnvBase)
        self._opt_func = self.env.make_optimizable_func()
        self._inference_func = self.env.make_func()

    def _run_step(self, data_list):
        self._gen_advantage(data_list)
        feed_dict = self._get_feed_dict(data_list)

        if self._has_value_regressor():
            self._run_step_v_regressor(feed_dict)

        net_outputs = self._run_step_network(feed_dict)

        # Summaries
        avg_score = sum([data['score'] for data in data_list]) / len(data_list)
        output = dict(score=avg_score)
        summaries = tf.Summary()
        summaries.value.add(tag='train/score', simple_value=avg_score)

        if 'summaries' in net_outputs:
            summaries.value.MergeFrom(tf.Summary.FromString(net_outputs['summaries']).value)
            output.update(net_outputs)

        self.runtime['summaries'] = summaries
        return output

    def _run_step_network(self, data):
        raise NotImplementedError()

    def _initialize_summaries(self):
        raise NotImplementedError()

    def _compile_fn_train(self):
        raise NotImplementedError()


class TRPOTrainer(AlterSurrOptimizationTrainerBase):
    _p_feed_dict_keys = ['state', 'action', 'advantage', 'theta_old']

    def _initialize_summaries(self):
        with self.env.as_default():
            summary.scalar('policy_loss', self.env.policy_loss, collections=[ACGraphKeys.POLICY_SUMMARIES])
            summary.scalar('policy_kl', self.env.policy_kl, collections=[ACGraphKeys.POLICY_SUMMARIES])
            if not self._has_value_regressor():
                summary.scalar('value_loss', self.env.value_loss, collections=[ACGraphKeys.VALUE_SUMMARIES])

    def _compile_fn_train(self):
        self._p_func.compile([])
        self._compile_func_with_summary(
            self._p_func_inference, {'p_loss': self.env.policy_loss, 'p_kl': self.env.policy_kl},
            ACGraphKeys.POLICY_SUMMARIES)
        if self._value_regressor is None:
            self._compile_func_with_summary(
                self._v_func, {'v_loss': self.env.value_loss}, ACGraphKeys.VALUE_SUMMARIES)

    def _run_step_p(self, feed_dict):
        feed_dict = {k: feed_dict[k] for k in self._p_feed_dict_keys}
        self._p_func.call_args(feed_dict)
        p_outputs = self._p_func_inference.call_args(feed_dict)
        return p_outputs


class PPOTrainerMixin(ACOptimizationTrainerBase):
    _batch_sampler = None
    _inference_batch_size = None

    def _initialize_summaries(self):
        with self.env.as_default():
            summary.scalar('policy_loss', self.env.policy_loss, collections=[ACGraphKeys.POLICY_SUMMARIES])
            if not self._has_value_regressor():
                summary.scalar('value_loss', self.env.value_loss, collections=[ACGraphKeys.VALUE_SUMMARIES])

    @notnone_property
    def batch_sampler(self):
        return self._batch_sampler

    def set_batch_sampler(self, sampler):
        self._batch_sampler = sampler

    def inference_batch_size(self):
        return self._inference_batch_size

    def _make_inference_batch(self, feed_dict):
        if self._inference_batch_size is None:
            return feed_dict
        else:
            idx = random.randint(nd_batch_size(feed_dict), size=self._inference_batch_size)
            return {k: v[idx] for k, v in feed_dict.items()}

    def set_inference_batch_size(self, batch_size):
        self._inference_batch_size = batch_size


class PPOTrainer(PPOTrainerMixin, AlterSurrOptimizationTrainerBase):
    _p_feed_dict_keys = ['state', 'action', 'advantage', 'theta_old']

    def _compile_fn_train(self):
        self._p_func.compile([])
        self._compile_func_with_summary(
            self._p_func_inference, {'p_loss': self.env.policy_loss},
            ACGraphKeys.POLICY_SUMMARIES)

        if self._value_regressor is None:
            self._compile_func_with_summary(
                self._v_func, {'v_loss': self.env.value_loss}, ACGraphKeys.VALUE_SUMMARIES)

    def _run_step_p(self, feed_dict):
        iterator = self.batch_sampler(feed_dict, self._p_feed_dict_keys)
        for batch in tqdm(
                iterator,
                desc='Proximal policy optimizing',
                total=len(iterator),
                leave=False,
                **get_tqdm_defaults()
            ):

            self._p_func.call_args(batch)
        p_outputs = self._p_func_inference.call_args(
            self._make_inference_batch({k: feed_dict[k] for k in self._p_feed_dict_keys})
        )
        return p_outputs

    def _run_step_v_network(self, feed_dict):
        iterator = self.batch_sampler(feed_dict, ['state', 'return_'], renames=['state', 'value_label'])
        for batch in tqdm(
                iterator,
                desc='Proximal value optimizing',
                total=len(iterator),
                leave=False,
                **get_tqdm_defaults()
            ):

            self._v_func.call_args(batch)
        return None


class PPOTrainerV2(PPOTrainerMixin, JointSurrOptimizationTrainerBase):
    _feed_dict_keys = JointSurrOptimizationTrainerBase._feed_dict_keys + ['value', 'return_']
    _p_feed_dict_keys = ['state', 'action', 'advantage', 'theta_old', 'value', 'return_']
    _p_feed_dict_renames = ['state', 'action', 'advantage', 'theta_old', 'value_old', 'value_label']

    def _compile_fn_train(self):
        self._opt_func.compile([])
        self._compile_func_with_summary(self._inference_func,
                                        {'p_loss': self.env.policy_loss, 'v_loss': self.env.value_loss})

    def _run_step_network(self, feed_dict):
        iterator = self.batch_sampler(feed_dict, self._p_feed_dict_keys, renames=self._p_feed_dict_renames)
        for batch in tqdm(
                iterator,
                desc='Proximal policy optimizing',
                total=len(iterator),
                leave=False,
                **get_tqdm_defaults(),
            ):

            self._opt_func.call_args(batch)
        opt_outputs = self._inference_func.call_args(self._make_inference_batch({
            k1: feed_dict[k2] for k1, k2 in zip(self._p_feed_dict_renames, self._p_feed_dict_keys)
        }))
        return opt_outputs

