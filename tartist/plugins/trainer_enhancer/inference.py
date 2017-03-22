# -*- coding:utf8 -*-
# File   : inference.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/1/17
# 
# This file is part of TensorArtist


from .summary import put_summary_history
from tartist.core import register_event
from tartist.nn import TArtGraphKeys
import tensorflow as tf
import tqdm as tqdm


def enable_inference_runner(trainer, dataflow, interval=1,
                            extra_outputs=None, extra_outputs_callback=None, *,
                            run_on_epoch0=False, collection_key=TArtGraphKeys.INFERENCE_SUMMARIES):

    extra_outputs = extra_outputs or {}

    def compile_fn_inference(trainer):
        trainer._fn_inference = func = trainer.env.make_func()
        summaries = trainer.network.get_merged_summaries(collection_key)
        if summaries is not None:
            func.add_extra_kwoutput('summaries', summaries)
        func.compile(extra_outputs)
        if func.queue_enabled:
            func.disable_queue()

    def run_step(trainer):
        epoch = trainer.epoch
        if not ((epoch == 0 and run_on_epoch0) or (epoch != 0 and epoch % interval == 0)):
            return

        if hasattr(trainer, '_iter_inference'):
            df = trainer._iter_inference
        else:
            df = dataflow(trainer.env)
            trainer._iter_inference = df

        expect_count = 0
        try:
            expect_count = len(df)
            if expect_count is None:
                expect_count = 0
        except:
            pass

        df = tqdm.tqdm(df, total=expect_count, leave=False, desc='running inference')
        count = 0
        for data in df:
            count += 1
            out = trainer._fn_inference.call_args(data)
            if 'summaries' in out:
                summaries = tf.Summary.FromString(out['summaries'])
                put_summary_history(trainer, summaries)
            if extra_outputs_callback:
                extra_outputs_callback(trainer, data, out)
        trainer.runtime['inference_epoch_size'] = count

    register_event(trainer, 'initialization:after', compile_fn_inference)
    register_event(trainer, 'epoch:after', run_step, priority=5)
