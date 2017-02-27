# -*- coding:utf8 -*-
# File   : progress.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/26/17
# 
# This file is part of TensorArtist

from tartist.core import register_event
import tqdm


def enable_epoch_progress(trainer):
    pbar = None

    def epoch_progress_on_epoch_before(trainer):
        nonlocal pbar
        pbar = tqdm.tqdm(total=trainer.epoch_size, leave=False)

    def epoch_progress_on_iter_after(trainer, inp, out):
        nonlocal pbar
        pbar.update()
        pbar.set_description('Iter#{}: loss={:.4f}'.format(trainer.iter, trainer.runtime.get('loss', 0)))

    def epoch_progress_on_epoch_after(trainer):
        nonlocal pbar
        pbar.close()

    register_event(trainer, 'epoch:before', epoch_progress_on_epoch_before, priority=25)
    register_event(trainer, 'iter:after', epoch_progress_on_iter_after, priority=25)
    register_event(trainer, 'epoch:after', epoch_progress_on_epoch_after, priority=5)
