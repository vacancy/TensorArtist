# -*- coding:utf8 -*-
# File   : snapshot.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/26/17
# 
# This file is part of TensorArtist

from tartist.core import get_env, register_event, get_logger
from tartist.core import io
import os.path as osp
import numpy as np

logger = get_logger()

__snapshot_dir__ = 'snapshots'
__snapshot_ext__ = '.snapshot.pkl'
__weights_ext__ = '.weights.pkl'


def get_snapshot_dir():
    return get_env('dir.snapshot', osp.join(get_env('dir.root'), __snapshot_dir__))


def enable_snapshot_saver(trainer, save_interval=1):
    def dump_snapshot_on_epoch_after(trainer):
        if trainer.epoch % save_interval != 0:
            return

        snapshot_dir = get_snapshot_dir()
        snapshot = trainer.dump_snapshot()
        fpath = osp.join(snapshot_dir, 'epoch_{}'.format(trainer.epoch) + __snapshot_ext__)
        io.mkdir(osp.dirname(fpath))
        io.dump(fpath, snapshot)

        fpath_aliased = []

        fpath_last = osp.join(snapshot_dir, 'last_epoch' + __snapshot_ext__)
        fpath_aliased.append(fpath_last)

        best_loss = trainer.runtime.get('best_loss', np.inf)
        current_loss = trainer.runtime.get('loss', None)
        if current_loss is not None and best_loss > current_loss:
            trainer.runtime['best_loss'] = current_loss

            fpath_best_loss = osp.join(snapshot_dir, 'best_loss' + __snapshot_ext__)
            fpath_aliased.append(fpath_best_loss)

        best_error = trainer.runtime.get('best_error', np.inf)
        current_error = trainer.runtime.get('error', None)
        if current_error is not None and best_error > current_error:
            trainer.runtime['best_error'] = current_error

            fpath_best_error = osp.join(snapshot_dir, 'best_error' + __snapshot_ext__)
            fpath_aliased.append(fpath_best_error)

        io.link(fpath, *fpath_aliased)

        logger.info('Model at epoch {} dumped to {}.\n(Also alias: {})'.format(
            trainer.epoch, fpath, ', '.join(fpath_aliased)))

    register_event(trainer, 'epoch:after', dump_snapshot_on_epoch_after, priority=20)


def load_snapshot_file(trainer, fpath):
    fpath = io.assert_extension(fpath, __snapshot_ext__)
    snapshot = io.load(fpath)
    if snapshot is None:
        return False
    trainer.load_snapshot(snapshot)
    return True


def load_weights_file(env, fpath):
    weights = io.load(fpath)
    if weights is None:
        return False

    if fpath.endswith(__snapshot_ext__):
        weights = weights['variables']
    env.network.assign_all_variables_dict(weights)
    return True


def dump_weights_file(env, fpath):
    fpath = io.assert_extension(fpath, __weights_ext__)
    weights = env.network.fetch_all_variables_dict()
    io.dump(fpath, weights)
    return fpath


def enable_snapshot_loading_after_initialization(trainer, *, continue_last=None, continue_from=None):
    assert continue_last is None or continue_from is None

    def load_snapshot_on_initialization_after(trainer):
        snapshot_dir = get_snapshot_dir()

        fpath = None
        if continue_last:
            fpath = osp.join(snapshot_dir, 'last_epoch' + __snapshot_ext__)
        if continue_from:
            fpath = osp.join(snapshot_dir, 'epoch_{}'.format(continue_from) + __snapshot_ext__)
        if fpath:
            if load_snapshot_file(trainer, fpath):
                fpath_real = osp.relpath(osp.realpath(fpath), osp.dirname(fpath))
                logger.info('Restored snapshot from {} (aka. {}), continue={}'.format(
                    fpath, fpath_real, continue_last, continue_from))

    register_event(trainer, 'initialization:after', load_snapshot_on_initialization_after, priority=25)


def enable_weights_loading_after_intialization(trainer, weights_fpath):
    def load_weights_on_initialization_after(trainer):
        if load_weights_file(trainer.env, weights_fpath):
            logger.info('Restored weights from {}'.format(weights_fpath))

    register_event(trainer, 'initialization:after', load_weights_on_initialization_after, priority=25)
