# -*- coding:utf8 -*-
# File   : lmdb.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/30/17
# 
# This file is part of TensorArtist

from .base import KVStoreBase
from ...core.utils.cache import cached_property

import os
import lmdb
import pickle

__all__ = ['LMDBKVStore']

_loads = pickle.loads
_dumps = pickle.dumps


class LMDBKVStore(KVStoreBase):
    def __init__(self, lmdb_path, readonly=True):
        super().__init__(readonly=readonly)
        self._lmdb = lmdb.open(lmdb_path,
                               subdir=os.path.isdir(lmdb_path),
                               readonly=True,
                               lock=False,
                               readahead=False,
                               map_size=1099511627776 * 2,
                               max_readers=100)

    @cached_property
    def txn(self):
       return self._lmdb.begin(write=not self.readonly)

    def _get(self, key, default):
        value = self.txn.get(key.encode('ascii'), default=default)
        value = pickle.loads(value)
        return value

    def _put(self, key, value, replace=False):
        self.txn.put(key.encode('ascii'), pickle.dumps(value), overwrite=replace)

    def _transaction(self, *args, **kwargs):
        return self.txn
