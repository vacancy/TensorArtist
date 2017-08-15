# -*- coding:utf8 -*-
# File   : adv_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 15/08/2017
# 
# This file is part of TensorArtist.

from ..math_utils import discount_cumsum, compute_gae


class AdvantageComputerBase(object):
    def __call__(self, data):
        self._compute(data)

    def _compute(self, data):
        raise NotImplementedError()


class DiscountedAdvantageComputer(AdvantageComputerBase):
    def __init__(self, gamma):
        self._gamma = gamma

    def _compute(self, data):
        return_ = discount_cumsum(data['reward'], self._gamma)
        advantage = return_ - data['value']

        data['return_'] = return_
        data['advantage'] = advantage


class GAEComputer(AdvantageComputerBase):
    def __init__(self, td, gae):
        self._td = td
        self._gae = gae

    def _compute(self, data):
        return_ = discount_cumsum(data['reward'], self._td)
        advantage = compute_gae(data['reward'], data['value'], 0, self._td, self._gae)

        data['return_'] = return_
        data['advantage'] = advantage
