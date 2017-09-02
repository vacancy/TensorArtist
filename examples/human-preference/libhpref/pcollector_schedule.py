# -*- coding:utf8 -*-
# File   : pcollector_schedule.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 22/08/2017
# 
# This file is part of TensorArtist.


class CollectorScheduler(object):
    def __init__(self, nr_total, nr_pretrain, nr_epochs):
        self._nr_total = nr_total
        self._nr_pretrain = nr_pretrain
        self._nr_epochs = nr_epochs

    def get_target(self, epoch):
        if epoch == 0:
            return self._nr_pretrain
        else:
            return self._get_annealed_target(epoch)

    def _get_annealed_target(self, epoch):
        raise NotImplementedError()


class LinearCollectorScheduler(CollectorScheduler):
    def _get_annealed_target(self, epoch):
        return int((self._nr_total - self._nr_pretrain) / self._nr_epochs * epoch) + self._nr_pretrain


class ExponentialDecayCollectorScheduler(CollectorScheduler):
    @property
    def _get_annealed_target(self, epoch):
        """Return the number of labels desired at this point in training. """
        exp_decay_frac = 0.01 ** (epoch / self._nr_epochs)  # Decay from 1 to 0
        pretrain_frac = self._nr_pretrain/ self._nr_total
        desired_frac = pretrain_frac + (1 - pretrain_frac) * (1 - exp_decay_frac)  # Start with 0.25 and anneal to 0.99
        return desired_frac * self._nr_total
