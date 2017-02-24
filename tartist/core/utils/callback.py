# -*- coding:utf8 -*-
# File   : callback.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
# 
# This file is part of TensorArtist

__all__ = ['CallbackManager']


class CallbackManager(object):
    """
    A callable manager utils.

    Using register(name, callback) to register a callback.

    Using dispatch(name, *args, **kwargs) to dispatch.


    If there exists a super callback, it will block all callbacks.
    A super callback will receive the called name as its first argument.

    Then the dispatcher will try to call the callback by name.
    If such name does not exists, a fallback callback will be called.

    The fallback callback will also receive the called name as its first argument.
    """

    def __init__(self):
        super().__init__()
        self._super_callback = None
        self._callbacks = dict()
        self._fallback_callback = None

    def register(self, name, callback):
        """
        Register a callable with (name, callback)
        :param name: the name
        :param callback: the callback
        :return: self
        """
        self._callbacks[name] = callback
        return self

    def get_callback(self, name):
        """
        Get a callable by name. If not exists, return None.
        :param name: the name
        :return: callable / None
        """
        if name in self._callbacks:
            return self._callbacks[name]
        return None

    def has_callback(self, name):
        """
        Tell whether there exists a callable of given name.
        :param name: the name
        :return: whether the callable exists
        """
        return name in self._callbacks

    def get_super_callback(self):
        """
        :return: the super callback
        """
        return self._super_callback

    def set_super_callback(self, callback):
        """
        :param callback: the new super callback
        :return: self
        """
        self._super_callback = callback
        return self

    def get_fallback_callback(self):
        """
        :return: the super callback
        """
        return self._fallback_callback

    def set_fallback_callback(self, callback):
        """
        :param callback: the new fallback callback
        :return: self
        """
        self._fallback_callback = callback
        return self

    def dispatch(self, name, *args, **kwargs):
        """
        Dispatch by name.
        :param name: the name
        :return: the result
        """
        if self._super_callback is not None:
            return self._super_callback(name, *args, **kwargs)
        return self.dispatch_direct(name, *args)

    def dispatch_direct(self, name, *args, **kwargs):
        """
        Dispatch by name, ignoring the super callback.
        This method is useful if you want to register a super callback.
        :param name: the name
        :return: the result
        """
        if name in self._callbacks:
            return self._callbacks[name](*args, **kwargs)
        elif self._fallback_callback is not None:
            return self._fallback_callback(name, *args, **kwargs)
        return None

