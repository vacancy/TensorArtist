# -*- coding:utf8 -*-
# File   : base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/30/17
# 
# This file is part of TensorArtist

__all__ = ['KVStoreBase', 'MemKVStore']

from ...core.utils.context import EmptyContext


class KVStoreBase(object):
    def __init__(self, readonly=False):
        self.__readonly = readonly

    @property
    def readonly(self):
        return self.__readonly

    def get(self, key, default=None):
        return self._get(self, key, default=default)

    def put(self, key, value, replace=True):
        assert not self.readonly, 'KVStore is readonly: {}'.format(self)
        return self._put(key, value, replace=replace)

    def transaction(self, *args, **kwargs):
        return self._transaction(*args, **kwargs)

    def keys(self):
        return self._keys()

    def _get(self, key, default):
        raise NotImplementedError()

    def _put(self, key, value, replace):
        raise NotImplementedError()

    def _transaction(self, *args, **kwargs):
        raise NotImplementedError()

    def _keys(self):
        assert False, 'KVStore does not support keys access'


class MemKVStore(KVStoreBase):
    def __init__(self, readonly=False):
        super().__init__(readonly=readonly)
        self._store = dict()

    def _get(self, key, default):
        return self._store.get(key, default)

    def _put(self, key, value, replace):
        if not replace:
            self._store.setdefault(key, value)
        else:
            self._store[key] = value

    def _transaction(self, *args, **kwargs):
        return EmptyContext()
    
    def _keys(self):
        return self._store.keys()

