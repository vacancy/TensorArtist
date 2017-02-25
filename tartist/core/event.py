# -*- coding:utf8 -*-
# File   : event.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
# 
# This file is part of TensorArtist

__all__ = [
    'EventManager', 
    'event_manager', 
    'register_event', 'unregister_event', 
    'trigger_event', 'trigger_event_args'
]

import uuid


class EventManager(object):
    DEF_PRIORITY = 10
    monitors = dict()
    ext_list = dict()

    def __init__(self):
        super().__init__()

    def register(self, point, name, callback, priority=DEF_PRIORITY, subkey=None):
        if isinstance(priority, str):
            subkey = priority
            priority = self.DEF_PRIORITY
        if subkey is None:
            subkey = uuid.uuid4() 

        self.monitors.setdefault(point, {}).setdefault(name, {}).setdefault(priority, {})[subkey] = callback

        return subkey

    def unregister(self, point, name, sign, priority=DEF_PRIORITY):
        if isinstance(sign, str):
            del self.monitors[point][name][priority][sign]
        else:
            pool = self.monitors[point][name][priority]
            for key in pool.keys():
                if pool[key] == sign:
                    del pool[key]

    def trigger(self, point, name, *args, **kwargs):
        self.trigger_args(point, name, args, kwargs)

    def trigger_args(self, point, name, args, kwargs):
        if point not in self.monitors:
            return
        if name not in self.monitors[point]:
            return
        if self.monitors[point][name] is None:
            return

        pools = self.monitors[point][name]
        for i in range(50):
            if i not in pools:
                continue
            for callback in pools[i].values():
                callback(*args, **kwargs)


event_manager = EventManager()
register_event = event_manager.register
unregister_event = event_manager.unregister
trigger_event = event_manager.trigger
trigger_event_args = event_manager.trigger_args

