# -*- coding:utf8 -*-
# File   : tfqueue.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/21/17
# 
# This file is part of TensorArtist

from .function import Function
from ..graph.node import as_tftensor
from ..tfutils import TArtGraphKeys

import threading
import tensorflow as tf
from tensorflow.contrib import graph_editor

__all__ = ['QueuedInputFunction', 'InputQueueDesc']


class QueuedInputFunction(Function):
    def __init__(self, env):
        super().__init__(env)
        self._input_queue_desc = env.input_queue_desc
        self._server_thread = None
        self._queue_enabled = True

    def enable_queue(self):
        self._queue_enabled = True

    def disable_queue(self):
        self._queue_enabled = False

    @property
    def queue_enabled(self):
        return self._queue_enabled

    def _server_mainloop(self, df):
        try:
            for feed_dict in df:
                for f in self._extra_kw_modifiers:
                    f(feed_dict)
                feed_dict = self.canonize_feed_dict(feed_dict)

                self.session.run(self._input_queue_desc.enqueue_op, feed_dict=feed_dict)
        except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
            try:
                self._env.session.run([self._input_queue_desc.close_op])
            except Exception:
                pass
            return
        except Exception as e:
            print("Exception in EnqueueThread:", e)

    def serve(self, dataflow):
        self._server_thread = threading.Thread(target=self._server_mainloop, args=(dataflow, ), daemon=True)
        self._server_thread.start()

    def __call__(self, *args, output_raw=False, **kwargs):
        if self._queue_enabled:
            assert len(args) == 0 and len(kwargs) == 0, 'Can not provide args for QueuedInputFunction'
            outputs = self.session.run(self._outputs)
            if output_raw:
                return outputs
            return self._output_manager.format(outputs)
        else:
            kwargs.setdefault(self._input_queue_desc.queue_cond.name, False)
            super().__call__(*args, output_raw=output_raw, **kwargs)


class InputQueueDesc(object):
    _placeholders = None
    _input_queue = None
    _input_queue_cond = None
    enqueue_op = None
    dequeue_op = None
    close_op = None
    qsize_op = None

    def __init__(self, env, name='input_queue'):
        self._name = name
        self._env = env
        self._placeholders = None

    def setup(self, graph):
        self._placeholders = graph.get_collection(TArtGraphKeys.PLACEHOLDERS)
        placeholders_dtypes = [x.dtype for x in self._placeholders]
        self._input_queue = tf.FIFOQueue(self._env.flags.input_queue_size, placeholders_dtypes, name=self._name)
        self._input_queue_cond = tf.placeholder_with_default(True, shape=[], name=self._name + '_cond')

        self.enqueue_op = self._input_queue.enqueue(self._placeholders)
        self.dequeue_op = self._input_queue.dequeue()
        self.close_op = self._input_queue.close(cancel_pending_enqueues=True)
        self.qsize_op = self._input_queue.size()

        for a, b in zip(self._placeholders, self.dequeue_op):
            as_tftensor(b).set_shape(as_tftensor(a).get_shape())

        self.edit_graph(graph)

    def edit_graph(self, graph):
        sgv0 = [as_tftensor(x) for x in self._placeholders]
        sgv1 = self.dequeue_op
        sgv2 = [tf.cond(self.queue_cond, lambda: x, lambda: y) for x, y in zip(sgv1, sgv0)]
        graph_editor.swap_ts(sgv0, sgv2, cannot_modify=[self.enqueue_op] + [x.op for x in sgv2])

    @property
    def queue(self):
        return self._input_queue

    @property
    def queue_cond(self):
        return self._input_queue_cond
