# -*- coding:utf8 -*-
# File   : saver.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/26/17
# 
# This file is part of TensorArtist

from tartist.core import get_env, register_event, get_logger
from tartist.core import io
import os.path as osp

logger = get_logger()

__snapshot_dir__ = 'snapshots'
__snapshot_ext__ = '.snapshot.pkl'


def set_model_error_getter(trainer, getter):
    trainer._model_error_getter = getter


def enable_model_saver(trainer, save_interval=1):
    def dump_snapshot_on_epoch_after(trainer):
        if trainer.epoch % save_interval != 0:
            return

        snapshot = trainer.dump_snapshot()
        fpath = osp.join(get_env('dir.root'), __snapshot_dir__, 'epoch_{}'.format(trainer.epoch) + __snapshot_ext__)
        io.mkdir(osp.dirname(fpath))
        io.dump(fpath, snapshot)
        fpath_last = osp.join(get_env('dir.root'), __snapshot_dir__, 'last_epoch' + __snapshot_ext__)
        io.link(fpath, fpath_last)

        logger.info('Model at epoch {} dumped to {}.\n(Also alias: {})'.format(trainer.epoch, fpath, fpath_last))

    register_event(trainer, 'epoch:after', dump_snapshot_on_epoch_after, priority=20)
