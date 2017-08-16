# -*- coding:utf8 -*-
# File   : progress.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/26/17
# 
# This file is part of TensorArtist.

from tartist.core.utils.thirdparty import get_tqdm_defaults
import tqdm
import numpy as np


def enable_epoch_progress(trainer):
    pbar = None

    def epoch_progress_on_iter_after(trainer, inp, out):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm.tqdm(total=trainer.epoch_size, leave=False, initial=trainer.iter % trainer.epoch_size,
                             **get_tqdm_defaults())

        desc = 'Iter={}'.format(trainer.iter)
        if 'error' in trainer.runtime:
            desc += ', error={:.4f}'.format(trainer.runtime['error'])
        for k in sorted(out.keys()):
            v = out[k]
            if isinstance(v, (str, int, float, np.ndarray, np.float32, np.float64, np.int32, np.int64)):
                try:
                    v = float(v)
                    desc += ', {}={:.4f}'.format(k, v)
                except ValueError:
                    pass
        pbar.set_description(desc)
        pbar.update()

    def epoch_progress_on_epoch_after(trainer):
        nonlocal pbar
        pbar.close()
        pbar = None

    trainer.register_event('iter:after', epoch_progress_on_iter_after, priority=25)
    trainer.register_event('epoch:after', epoch_progress_on_epoch_after, priority=5)
