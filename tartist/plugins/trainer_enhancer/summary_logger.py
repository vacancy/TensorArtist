# -*- coding:utf8 -*-
# File   : summary_logger.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/26/17
# 
# This file is part of TensorArtist

from tartist.core import get_logger, register_event

logger = get_logger()


class SummaryHistoryManager(object):
    def __init__(self):
        self._summaries = {}
        self._summaries_type = {}

    @property
    def all_summaries(self):
        return self.get_all_summaries()

    def get_all_summaries(self, type=None):
        if type is None:
            return list(self._summaries_type.keys())
        filt = lambda x: type == x
        return [k for k, v in self._summaries_type.items() if filt(v)]

    def clear_all(self):
        self._summaries = {}

    def clear(self, key):
        self._summaries[key] = []

    def put_scalar(self, key, value):
        value = float(value)
        self._summaries.setdefault(key, []).append(value)

    def put_summaries(self, summaries):
        for val in summaries.value:
            if val.WhichOneof('value') == 'simple_value':
                self.put_scalar(val.tag, val.simple_value)
                self.set_type(val.tag, 'scalar')

    def get(self, key):
        return self._summaries.get(key, [])

    def has(self, key):
        return key in self._summaries

    def get_type(self, key):
        return self._summaries_type.get(key, 'unknown')

    def set_type(self, key, value, check=True):
        old_value = self.get_type(key)
        if old_value != 'unknown' and check:
            assert old_value == value, 'summary type mismatched'
        self._summaries_type[key] = value

    def average(self, key, top_k=None):
        values = self._summaries.get(key, [])
        if top_k is None:
            top_k = len(values)
        values = values[-top_k:]
        return sum(values) / len(values)


def enable_summary_history(trainer):
    def check_proto_contains(proto, tag):
        if proto is None:
            return False
        for v in proto.value:
            if v.tag == tag:
                return True
        return False

    def summary_history_on_optimization_before(trainer):
        trainer.runtime['summary_histories'] = SummaryHistoryManager()

    def summary_history_on_iter_after(trainer, inp, out):
        mgr = trainer.runtime['summary_histories']

        summaries = None
        if 'summaries' in trainer.runtime:
            summaries = trainer.runtime['summaries']
            mgr.put_summaries(summaries)
        if 'loss' in trainer.runtime and check_proto_contains(summaries, 'loss'):
            mgr.set_type('loss', 'scalar')
            mgr.put_scalar('loss', trainer.runtime['loss'])
        error_summary_key = trainer.runtime.get('error_summary_key', None)

        if mgr.has(error_summary_key):
            trainer.runtime['error'] = mgr.get(error_summary_key)[-1]
            if check_proto_contains(summaries, 'error'):
                mgr.set_type('error', 'scalar')
                mgr.put_scalar('error', trainer.runtime['error'])

    register_event(trainer, 'optimization:before', summary_history_on_optimization_before)
    register_event(trainer, 'iter:after', summary_history_on_iter_after)


def enable_echo_summary_scalar(trainer):
    def summary_history_scalar_on_epoch_after(trainer):
        mgr = trainer.runtime['summary_histories']

        log_strs = ['Summaries: epoch = {}'.format(trainer.epoch)]
        for k in mgr.get_all_summaries('scalar'):
            avg = mgr.average(k, trainer.epoch_size)
            log_strs.append('  {} = {}'.format(k, avg))
        if len(log_strs) > 1:
            logger.info('\n'.join(log_strs))

    register_event(trainer, 'epoch:after', summary_history_scalar_on_epoch_after)


def set_error_summary_key(trainer, key):
    trainer.runtime['error_summary_key'] = key

