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


def enable_model_saver(trainer, save_interval=1):
    def dump_snapshot_on_epoch_after(trainer):
        if trainer.epoch % save_interval != 0:
            return

        snapshot = trainer.dump_snapshot()
        fpath = osp.join(get_env('dir.root'), __snapshot_dir__, 'epoch_{}'.format(trainer.epoch) + __snapshot_ext__)
        io.mkdir(osp.dirname(fpath))
        io.dump(fpath, snapshot)

        fpath_aliased = []

        fpath_last = osp.join(get_env('dir.root'), __snapshot_dir__, 'last_epoch' + __snapshot_ext__)
        fpath_aliased.append(fpath_last)

        best_loss = trainer.runtime.get('best_loss', np.inf)
        current_loss = trainer.runtime.get('loss', None)
        if current_loss is not None and best_loss > current_loss:
            trainer.runtime['best_loss'] = current_loss

            fpath_best_loss = osp.join(get_env('dir.root'), __snapshot_dir__, 'best_loss' + __snapshot_ext__)
            fpath_aliased.append(fpath_best_loss)

        best_error = trainer.runtime.get('best_error', np.inf)
        current_error = trainer.runtime.get('error', None)
        if current_error is not None and best_error > current_error:
            trainer.runtime['best_error'] = current_error

            fpath_best_error = osp.join(get_env('dir.root'), __snapshot_dir__, 'best_error' + __snapshot_ext__)
            fpath_aliased.append(fpath_best_error)

        io.link(fpath, *fpath_aliased)

        logger.info('Model at epoch {} dumped to {}.\n(Also alias: {})'.format(
            trainer.epoch, fpath, ', '.join(fpath_aliased)))

    register_event(trainer, 'epoch:after', dump_snapshot_on_epoch_after, priority=20)
