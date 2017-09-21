# -*- coding:utf8 -*-
# File   : experience.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/08/2017
# 
# This file is part of TensorArtist.

from tartist.core import get_logger
from tartist.core.utils.meta import map_exec
from tartist.core.utils.concurrent_stat import TSCounterBasedEvent, TSCoordinatorEvent
from tartist.core.utils.thirdparty import get_tqdm_defaults
from tartist.data.flow import SimpleDataFlowBase

from threading import Thread
from tqdm import tqdm

import queue
import threading
import collections
import numpy as np

logger = get_logger(__file__)

__all__ = ['SynchronizedExperienceCollector', 'SynchronizedTrajectoryDataFlow']


Experience = collections.namedtuple('Experience', ('state', 'action', 'outputs', 'reward', 'is_over'))
Prediction = collections.namedtuple('Prediction', ('action', 'outputs'))


class PredictionFuture(object):
    def __init__(self):
        self._result = None
        self._event = threading.Event()

    def set_result(self, result):
        self._result = result
        self._event.set()

    def wait(self):
        self._event.wait()
        self._event.clear()
        return self._result


class SynchronizedExperienceCollector(object):
    """Synchronized experience collector."""

    def __init__(self, owner_env,
                 make_player, output2action,
                 nr_workers, nr_predictors,
                 mode='EPISODE',
                 predictor_output_names=None,
                 predictor_batch_size=16,
                 output2action_ts=True):
        """

        :param owner_env: owner environment for infer the agent.
        :param make_player: callable, return the environment (player).
        :param output2action: callable, network's output => action.
        :param nr_workers: number of environment workers.
        :param nr_predictors: number of predictors forwarding the neural network.
        :param mode: collection mode:
            - EPISODE: collect `target` episode
            - EPISODE: collect several episode until reach total steps `target`
            - STEP: collect `target` primitive steps
        :param predictor_output_names: predictor's output names, used for compose the function.
        :param predictor_batch_size: predictor's batch size, typically 4/8/16.
        """

        self._owner_env = owner_env
        self._make_player = make_player
        self._output2action = output2action
        if output2action_ts:
            self._output2action_mutex = threading.Lock()
        else:
            self._output2action_mutex = None

        self._nr_workers = nr_workers
        self._nr_predictors = nr_predictors
        self._mode = mode
        assert self._mode in ('EPISODE', 'EPISODE-STEP', 'STEP')

        self._predictor_output_names = predictor_output_names
        self._predictor_batch_size = predictor_batch_size

        self._task_start = TSCoordinatorEvent(self._nr_workers)
        self._task_end = TSCoordinatorEvent(self._nr_workers)

        self._prediction_queue = queue.Queue()
        self._trajectories = []
        self._trajectories_counter = None
        self._collect_mutex = threading.Lock()

    @property
    def owner_env(self):
        return self._owner_env

    @property
    def mode(self):
        return self._mode

    def initialize(self):
        workers = [Thread(target=self._worker_thread, args=(i, ), daemon=True) for i in range(self._nr_workers)]
        predictors = [Thread(target=self._predictor_thread, daemon=True) for i in range(self._nr_predictors)]

        map_exec(Thread.start, workers)
        map_exec(Thread.start, predictors)

    def collect(self, target):
        with self._collect_mutex:
            return self.__collect(target)

    def __collect(self, target):
        self._trajectories = [[] for _ in range(self._nr_workers)]
        self._trajectories_counter = TSCounterBasedEvent(
                target, tqdm=tqdm(total=target, leave=False, desc='Trajectory collecting', **get_tqdm_defaults()))

        # Start all workers.
        self._task_start.broadcast()

        self._trajectories_counter.wait()
        self._trajectories_counter.clear()

        # Acquire stop.
        self._task_end.broadcast()

        # Reduce all outputs.
        if self._mode.startswith('EPISODE'):
            outputs = []
            for ts in self._trajectories:
                outputs.extend(ts)
            return outputs
        else:
            return self._trajectories.copy()

    def _worker_thread(self, worker_id):
        player = self._make_player()
        future = PredictionFuture()

        while True:
            self._task_start.wait()
            player.restart()

            this_episode = []
            if self._mode.startswith('EPISODE'):
                self._trajectories[worker_id].append(this_episode)

            while True:
                if self._task_end.check():
                    break

                state = player.current_state
                self._prediction_queue.put((state, future))
                prediction = future.wait()
                reward, is_over = player.action(prediction.action)
                exp = Experience(state, prediction.action, prediction.outputs, reward, is_over)

                if self._mode.startswith('EPISODE'):
                    this_episode.append(exp)
                    if self._mode == 'EPISODE-STEP':
                        self._trajectories_counter.tick()
                else:
                    self._trajectories[worker_id].append(exp)
                    self._trajectories_counter.tick()

                if is_over:
                    player.restart()

                    this_episode = []
                    if self._mode.startswith('EPISODE'):
                        self._trajectories[worker_id].append(this_episode)
                    if self._mode == 'EPISODE':
                        self._trajectories_counter.tick()

    def _predictor_thread(self):
        batch_size = self._predictor_batch_size
        func = self._owner_env.make_func()
        self._compile_predictor_func(func)

        while True:
            nr_total = 0
            batched = []
            futures = []

            for i in range(batch_size):
                if i == 0:
                    state, future = self._prediction_queue.get()
                else:
                    try:
                        state, future = self._prediction_queue.get_nowait()
                    except queue.Empty:
                        break

                batched.append(state)
                futures.append(future)
                nr_total += 1

            self._infer_predictor_func(func, batched, futures, nr_total)

    def _compile_predictor_func(self, func):
        if self._predictor_output_names is None:
            func.compile(self._owner_env.network.outputs)
        else:
            func.compile({k : self._owner_env.network.outputs[k] for k in self._predictor_output_names})

    def _infer_predictor_func(self, func, batched, futures, nr_total):
        batched = np.array(batched)
        outputs = func(state=batched)
        for i in range(nr_total):
            this_output = {k: v[i] for k, v in outputs.items()}
            action = self._output2action_wrapped(this_output)
            pred = Prediction(action, this_output)
            futures[i].set_result(pred)

    def _output2action_wrapped(self, output):
        if self._output2action_mutex is None:
            return self._output2action(output)
        else:
            with self._output2action_mutex:
                return self._output2action(output)


class SynchronizedTrajectoryDataFlow(SimpleDataFlowBase):
    def __init__(self, collector, target, incl_value=True):
        self._collector = collector
        self._target = target
        self._incl_value = incl_value

        assert self._collector.mode.startswith('EPISODE')

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


