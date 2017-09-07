# -*- coding:utf8 -*-
# File   : rpredictor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/07/2017
# 
# This file is part of TensorArtist.

from tartist import random
from tartist.core import get_logger, register_event
from tartist.core.utils.meta import map_exec
from tartist.core.utils.logging import EveryNSecondLogger
from tartist.data.flow import PoolRandomSampleDataFlow, PoolDataFlow
from tartist.nn.train import SimpleTrainerEnv, SimpleTrainer
import threading
import collections
import numpy as np

__all__ = ['TrainingData', 'PredictorDesc', 'EnsemblePredictor']

TrainingData = collections.namedtuple('TrainingData', ['t1_state', 't1_action', 't2_state', 't2_action', 'pref'])
PredictorDesc = collections.namedtuple('PredictorDesc', ['make_network', 'make_optimizer',
                                                         'wrap_dataflow_train', 'wrap_dataflow_validation',
                                                         'main_train'])
logger = get_logger(__file__)


def _list_choice_n(a, size, rng=None):
    res = []
    for i in range(size):
        res.append(random.list_choice(a, rng=rng))
    return res


def _compute_e_var(rs, ret_variance):
    e_r = sum(rs) / len(rs)

    if ret_variance:
        rs_sqr = map_exec(lambda x: x ** 2, rs)
        var_r = sum(rs_sqr) / len(rs_sqr) - e_r ** 2
        return e_r, var_r

    return e_r


def _default_main_train(trainer):
    # TODO:: Early stop
    def on_optimization_before(trainer):
        # clear the validation loss history
        trainer.runtime['validation_losses'] = []

        # compile the function for inference
        trainer.inference_func = trainer.env.make_func()
        trainer.inference_func.compile(trainer.network.loss)

    def on_epoch_after(trainer):
        # compute the validation loss
        sum_loss, nr_data = 0, 0
        for data in trainer.dataflow_validation:
            sum_loss += trainer.inference_func(**data)
            nr_data += 1
        avg_loss = sum_loss / nr_data
        logger.info('Epoch: {}: average validation loss = {}.'.format(trainer.epoch, avg_loss))

        trainer.runtime['validation_losses'].append(avg_loss)

        # test whether early stop
        losses = trainer.runtime['validation_losses'][-6:]
        if len(losses) <= 1:
            return

        # 2 out of 5
        nr_loss_increase = 0
        for a, b in zip(losses[:-1], losses[1:]):
            if b > a:
                nr_loss_increase += 1

        if nr_loss_increase >= 2:
            # acquire early stop
            logger.critical('Validation loss is keeping increasing: acquire early stop.')
            trainer.stop()

    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer)
    summary.enable_echo_summary_scalar(trainer, enable_json=False, enable_tensorboard=False)

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    # early stop related hooks
    trainer.register_event('optimization:before', on_optimization_before)
    trainer.register_event('epoch:after', on_epoch_after)

    trainer.train()


def _default_wrap_dataflow(env, df):
    return df


class PredictorBase(object):
    def initialize(self):
        pass

    def add_training_data(self, data):
        raise NotImplementedError()

    def extend_training_data(self, data):
        raise NotImplementedError()

    def predict(self, state, action):
        raise NotImplementedError()

    def predict_batch(self, state_batch, action_batch):
        raise NotImplementedError()

    def ready_for_step(self, epoch):
        raise NotImplementedError()


class EnsemblePredictor(PredictorBase):
    _validation_ratio = 1 / 2.718281828
    _network_output_name = 'reward'

    def __init__(self, owner_env, scheduler, desc, nr_ensembles, devices,
                 nr_epochs, epoch_size, retrain_thresh=10):

        self._owner_env = owner_env
        self._scheduler = scheduler
        self._schedule_logger = EveryNSecondLogger(logger, 2)
        self._desc = desc
        self._nr_ensembles = nr_ensembles
        self._devices = devices
        self._nr_epochs = nr_epochs
        self._epoch_size = epoch_size
        self._retrain_thresh = retrain_thresh

        self._envs = []
        self._envs_predict = []
        self._funcs_predict = []
        self._funcs_predict_lock = threading.Lock()
        self._dataflows = []

        self._data_pool = []
        # number of data points used for training last time step
        self._data_pool_last = 0
        self._data_pool_lock = threading.Lock()
        # list of list of data
        self._training_sets = []
        # list of data
        self._validation_set = []

        self._training_lock = threading.Lock()

        self._rng = random.gen_rng()

    def initialize(self):
        self.__initialize_envs()

    def add_training_data(self, data, acquire_lock=True, try_retrain=True):
        assert isinstance(data, TrainingData)
        if data.pref == -1:
            return

        data = dict(t1_state=data.t1_state, t2_state=data.t2_state,
                    t1_action=data.t1_action, t2_action=data.t2_action,
                    pref=data.pref)

        if acquire_lock:
            self._data_pool_lock.acquire()
        self._data_pool.append(data)
        last_size, new_size = self._data_pool_last, len(self._data_pool)
        if acquire_lock:
            self._data_pool_lock.release()

        if try_retrain:
            if new_size - last_size > self._retrain_thresh:
                self._try_train_again()

    def extend_training_data(self, data, force_retrain=True):
        with self._data_pool_lock:
            for d in data[:-1]:
                self.add_training_data(d, acquire_lock=False, try_retrain=False)
            self.add_training_data(data[-1], acquire_lock=False, try_retrain=not force_retrain)

        if force_retrain:
            self._try_train_again()

    def predict(self, state, action, ret_variance=False):
        action = np.array(action)

        with self._funcs_predict_lock:
            if len(self._funcs_predict) == 0:
                rs = self._rng.random_sample(size=self._nr_ensembles)
            else:
                rs = []
                for f in self._funcs_predict:
                    r = f(state=state[np.newaxis], action=action[np.newaxis])[0]
                    rs.append(r)

            return _compute_e_var(rs, ret_variance=ret_variance)

    def predict_batch(self, state_batch, action_batch, ret_variance=False):
        with self._funcs_predict_lock:
            if len(self._funcs_predict) == 0:
                rs = self._rng.random_sample(size=(len(state_batch), self._nr_ensembles))
            else:
                rs = []
                for f in self._funcs_predict:
                    r = f(state=state_batch, action=action_batch)
                    rs.append(r)

            result = [_compute_e_var(r, ret_variance=ret_variance) for r in rs]
            return result

    def ready_for_step(self, epoch):
        with self._data_pool_lock:
            target, curr = self._scheduler.get_target(epoch), self._data_pool_last
        if curr < target:
            self._schedule_logger.info('Insufficient data for epoch {}: {}/{}.'.format(epoch, curr, target))
            return False
        return True

    @property
    def _make_network(self):
        return self._desc.make_network

    @property
    def _make_optimizer(self):
        return self._desc.make_optimizer

    @property
    def _wrap_dataflow_train(self):
        return self._desc.wrap_dataflow_train or _default_wrap_dataflow

    @property
    def _wrap_dataflow_validation(self):
        return self._desc.wrap_dataflow_validation or _default_wrap_dataflow

    @property
    def _main_train(self):
        return self._desc.main_train or _default_main_train

    def __initialize_envs(self):
        # Using rolling array
        def gen(i):
            for _ in range(2):
                env = SimpleTrainerEnv(SimpleTrainerEnv.Phase.TRAIN, self._devices[i])

                with env.as_default():
                    self._make_network(env)
                    self._make_optimizer(env)

                # initialize the fully random weights
                env.initialize_all_variables()
                env.share_func_lock_with(self._owner_env)
                yield env

        for eid in range(self._nr_ensembles):
            env, env_predict = gen(eid)
            self._envs.append(env)
            self._envs_predict.append(env_predict)

        with self._funcs_predict_lock:
            self.__make_predictors()

    def train_again(self):
        return self._try_train_again()

    def _try_train_again(self):
        """Try to run the training. If the predictors are being trained, do nothing"""
        rc = self._training_lock.acquire(blocking=False)

        if rc:
            t = threading.Thread(target=self.__do_train_again)
            t.start()
            self._training_lock.release()

        return rc

    def __do_train_again(self):
        """Do the actual training."""
        logger.critical('Predictors training begins.')
        with self._data_pool_lock:
            self.__split_training_data()

        # MJY(20170802):: It seems that we can not jointly run prediction and training even if
        # we already use rolling arrays due to some tensorflow bugs.
        # So here during training, we also acquire the lock.
        with self._funcs_predict_lock:
            self.__train()
            self.__make_predictors()
        logger.critical('Predictors training ends.')

    def __split_training_data(self):
        """Training step 1: split the training set and validation set."""

        self._data_pool_last = len(self._data_pool)
        # split the training set and validation set
        nr_validations = max(int(self._validation_ratio * len(self._data_pool)), 1)
        nr_training = max(len(self._data_pool) - nr_validations, 1)

        random.list_shuffle(self._data_pool, rng=self._rng)
        self._validation_set = self._data_pool[-nr_validations:]

        self._training_sets = []
        for i in range(self._nr_ensembles):
            self._training_sets.append(_list_choice_n(self._data_pool[:nr_training], nr_training, rng=self._rng))

        # make dataflows
        self._dataflows = []
        for i in range(self._nr_ensembles):
            df_train = PoolRandomSampleDataFlow(self._training_sets[i])
            df_train = self._wrap_dataflow_train(self._envs[i], df_train)
            df_validation = PoolDataFlow(self._validation_set)
            df_validation = self._wrap_dataflow_validation(self._envs[i], df_validation)
            self._dataflows.append((df_train, df_validation))

    def __train(self):
        """Training step 2: run the trainers."""

        logger.critical('Predictor ensemble retraining started.')

        for i in range(self._nr_ensembles):
            self.__train_thread(i)

        # all_threads = []
        # for i in range(self._nr_ensembles):
        #     t = threading.Thread(target=self.__train_thread, args=(i, ))
        #     all_threads.append(t)
        # map_exec(threading.Thread.start, all_threads)
        # map_exec(threading.Thread.join, all_threads)

        logger.critical('Predictor ensemble retraining finished.')

    def __train_thread(self, i):
        logger.info('Starting training for predictor #{}'.format(i))

        nr_iters = self._epoch_size * self._nr_epochs
        trainer = SimpleTrainer(nr_iters, env=self._envs[i],
                                data_provider=lambda e: self._dataflows[i][0], desc=self._desc)
        trainer.set_epoch_size(self._epoch_size)
        trainer.dataflow_validation = self._dataflows[i][1]

        # When you run train(), the parameters will be reset due to the calling of initialize_all_variables.
        self._main_train(trainer)

    def __make_predictors(self):
        """Training step 3: swap the envs for training and prediction, also build the function for prediction."""
        self._envs_predict, self._envs = self._envs, self._envs_predict
        self._funcs_predict = []
        for i, e in enumerate(self._envs_predict):
            f = e.make_func()
            f.compile(e.network.outputs[self._network_output_name])
            self._funcs_predict.append(f)
