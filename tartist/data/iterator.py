# -*- coding:utf8 -*-
# File   : iterator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/30/17
# 
# This file is part of TensorArtist

import numpy as np

from ..core.utils.nd import nd_batch_size

__all__ = [
    'DataIterator',
    'SingleIterator', 'NullIterator',
    'CountBasedIterator', 'ArrayIterator',
    'ListOfArrayIterator', 'DictOfArrayIterator'
]


class DataIterator(object):
    def __init__(self):
        self._is_first_iter = True

    def __len__(self):
        return self._count()

    def __iter__(self):
        self._initialize()
        self._is_first_iter = True
        return self

    def __next__(self):
        if not self._is_first_iter:
            if self._have_next():
                self._move_next()
            else:
                self._finalize()
                raise StopIteration()
        else:
            self._is_first_iter = False
        result = self._get()
        return result

    def _initialize(self):
        raise NotImplementedError()

    def _finalize(self):
        pass

    def _get(self):
        raise NotImplementedError()

    def _count(self):
        raise NotImplementedError()

    def _move_next(self):
        raise NotImplementedError()

    def _have_next(self):
        raise NotImplementedError()


class SingleIterator(DataIterator):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def _initialize(self):
        pass

    def _get(self):
        return self.data

    def _count(self):
        return 1

    def _move_next(self):
        pass

    def _have_next(self):
        return False


class NullIterator(DataIterator):
    def __iter__(self):
        pass

    def __next__(self):
        raise StopIteration()

    def _initialize(self):
        pass

    def _get(self):
        raise ValueError()

    def _count(self):
        return 0

    def _move_next(self):
        pass

    def _have_next(self):
        return False


class CountBasedIterator(DataIterator):
    def __init__(self, nr_batches):
        super().__init__()
        self._nr_batches = nr_batches
        self._current = 0

    def _initialize(self):
        self._current = 0

    def _get(self):
        return self._get_batch(self._current)

    def _get_batch(self, idx):
        raise NotImplementedError()

    def _count(self):
        return self._nr_batches

    def _move_next(self):
        self._current += 1

    def _have_next(self):
        return self._current < self._nr_batches - 1


class BatchBasedIterator(DataIterator):
    def __init__(self, length, batch_size, use_all_data=False):
        super().__init__()
        self._length = length
        self._batch_size = batch_size
        self._nr_batches = length // batch_size
        self._use_all_data = use_all_data
        if self.has_nonfull_batch:
            self._nr_batches += 1
        self._current = 0

    @property
    def has_nonfull_batch(self):
        return self._use_all_data and self._length % self._batch_size != 0

    def _initialize(self):
        self._current = 0

    def _get(self):
        if self._current == self._nr_batches - 1 and self.has_nonfull_batch:
            return self._get_batch(self._current, is_last=True)
        else:
            return self._get_batch(self._current, is_last=False)

    def _get_batch(self, idx, is_last=False):
        raise NotImplementedError()

    def _count(self):
        return self._nr_batches

    def _move_next(self):
        self._current += 1

    def _have_next(self):
        return self._current < self._nr_batches - 1


class ArrayIterator(BatchBasedIterator):
    def __init__(self, array, batch_size, use_all_data=False):
        super().__init__(len(array), batch_size, use_all_data)
        self._array = array

    def _get_batch(self, idx, is_last=False):
        if is_last:
            return np.array(self._array[idx * self._batch_size:])
        return np.array(self._array[idx * self._batch_size:(idx + 1) * self._batch_size])


class ListOfArrayIterator(BatchBasedIterator):
    def __init__(self, arrlist, batch_size, use_all_data=False):
        super().__init__(nd_batch_size(arrlist), batch_size, use_all_data)
        self._arrlist = arrlist

    def _get_batch(self, idx, is_last=False):
        start = idx * self._batch_size
        end = idx * self._batch_size if not is_last else self._length
        return [np.array(x[start:end]) for x in self._arrlist]


class DictOfArrayIterator(BatchBasedIterator):
    def __init__(self, arrdict, batch_size, use_all_data=False):
        super().__init__(nd_batch_size(arrdict), batch_size, use_all_data)
        self._arrdict = arrdict

    def _get_batch(self, idx, is_last=False):
        start = idx * self._batch_size
        end = (idx + 1) * self._batch_size if not is_last else self._length
        res = {k: np.array(v[start:end]) for k, v in self._arrdict.items()}
        return res

