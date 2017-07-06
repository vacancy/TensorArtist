# -*- coding:utf8 -*-
# File   : train.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/07/2017
# 
# This file is part of TensorArtist.

from tartist.app.rl.train import A3CMaster, A3CTrainer, A3CTrainerEnv
from tartist.core import get_env

import threading

__all__ = ['HPA3CMaster', 'HPA3CTrainerEnv', 'HPA3CTrainer']


class HPA3CMaster(A3CMaster):
    rpredictor = None

    # MJY(20170706): Add rpredictor access. Keep original version due to backward-compatibility.
    def _make_predictor_thread(self, i, func, daemon=True):
        return threading.Thread(target=self.predictor_func, daemon=daemon,
                                args=(i, self.rpredictor, self.router, self.queue, func))


class HPA3CTrainerEnv(A3CTrainerEnv):
    def _initialize_a3c_master(self):
        nr_predictors = get_env('a3c.nr_predictors')
        self._player_master = HPA3CMaster(self, 'hpa3c-player', nr_predictors)
        self._inference_player_master = HPA3CMaster(self, 'hpa3c-inference-player', nr_predictors)


class HPA3CTrainer(A3CTrainer):
    pass
