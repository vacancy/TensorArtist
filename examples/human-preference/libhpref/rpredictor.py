# -*- coding:utf8 -*-
# File   : rpredictor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/07/2017
# 
# This file is part of TensorArtist.

from tartist import random
from tartist.core.utils.meta import map_exec
from tartist.data.flow import LOARandomSampleDataFlow, ListOfArrayDataFlow
from tartist.nn.train import SimpleTrainerEnv, SimpleTrainer
import threading
import collections
import numpy as np

__all__ = ['TrainingData', 'PredictorDesc', 'EnsemblePredictor']

TrainingData = collections.namedtuple('TrainingData', ['t1_state', 't1_action', 't2_state', 't2_action', 'pref'])
PredictorDesc = collections.namedtuple('PredictorDesc', ['make_network', 'make_optimizer', 'wrap_dataflow_train',
                                                         'wrap_dataflow_validation', 'main_train'])


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
    trainer.train()


class EnsemblePredictor(object):
    _validation_ratio = 1 / 2.718281828
    _network_output_name = 'pred'

    def __init__(self, desc, nr_ensembles,
                 devices, nr_epochs, epoch_size, retrain_thresh=10):
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

    def add_data(self, data, acquire_lock=True, try_retrain=True):
        assert isinstance(data, TrainingData)
        if data.pref== -1:
            return

        p2 = data.pref
        p1 = 1 - p2
        data = dict(t1_state=data.t1_state, t2_state=data.t2_state,
                    t1_action=data.t1_action, t2_action=data.t2_action,
                    p1=p1, p2=p2)

        if acquire_lock:
            self._data_pool_lock.acquire()
        self._data_pool.append(data)
        last_size, new_size = self._data_pool_last, len(self._data_pool)
        if acquire_lock:
            self._data_pool_lock.release()

        if try_retrain:
            if new_size - last_size > self._retrain_thresh:
                self._try_train_again()

    def extend_data(self, data, force_retrain=True):
        with self._data_pool_lock:
            for d in data[:-1]:
                self.add_data(d, acquire_lock=False, try_retrain=False)
            self.add_data(data[-1], acquire_lock=False, try_retrain=not force_retrain)

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

    @property
    def _make_network(self):
        return self._desc.make_network

    @property
    def _make_optimizer(self):
        return self._desc.make_optimizer

    @property
    def _wrap_dataflow_train(self):
        return self._desc.wrap_dataflow_train

    @property
    def _wrap_dataflow_validation(self):
        return self._desc.wrap_dataflow_validation

    @property
    def _main_train(self):
        return self._desc.main_train or _default_main_train

    def __initialize_envs(self):
        def gen(i):
            for _ in range(2):
                env = SimpleTrainerEnv(SimpleTrainerEnv.Phase.TRAIN, self._devices[i])
                with env.as_default():
                    self._make_network(env)
                    self._make_optimizer(env)
                yield env

        for eid in range(self._nr_ensembles):
            env, env_predict = gen(eid)
            self._envs.append(env)
            self._envs_predict.append(env_predict)

    def __split_training_data(self):
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
            df_train = LOARandomSampleDataFlow(self._training_sets[i])
            df_train = self._wrap_dataflow_train(self._envs[i], df_train)
            df_validation = ListOfArrayDataFlow(self._validation_set)
            df_validation = self._wrap_dataflow_validation(self._envs[i], df_validation)
            self._dataflows.append((df_train, df_validation))

    def __train_thread(self, i):
        nr_iters = self._epoch_size * self._nr_epochs
        trainer = SimpleTrainer(nr_iters, env=self._envs[i],
                                data_provider=lambda e: self._dataflows[i][0], desc=self._desc)
        trainer.set_epoch_size(self._epoch_size)
        trainer.dataflow_validation = self._dataflows[i][1]

        # When you run train(), the parameters will be reset due to the calling of initialize_all_variables.
        self._main_train(trainer)

    def __train(self):
        all_threads = []
        for i in range(self._nr_ensembles):
            t = threading.Thread(target=self.__train_thread, args=(i, ))
            all_threads.append(t)
        map_exec(threading.Thread.start, all_threads)
        map_exec(threading.Thread.join, all_threads)

    def __make_predictors(self):
        self._envs_predict, self._envs = self._envs, self._envs_predict
        self._funcs_predict = []
        for i, e in enumerate(self._envs_predict):
            f = e.make_func()
            f.compile(e.network.outputs[self._network_output_name])
            self._funcs_predict.append(f)

    def __do_train_again(self):
        with self._data_pool_lock:
            self.__split_training_data()
        self.__train()
        with self._funcs_predict_lock:
            self.__make_predictors()

    def _try_train_again(self):
        rc = self._training_lock.acquire(blocking=False)

        if rc:
            t = threading.Thread(target=self.__do_train_again)
            t.start()
            self._training_lock.release()
