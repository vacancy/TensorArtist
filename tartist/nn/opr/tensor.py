# -*- coding:utf8 -*-
# File   : tensor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/17/17
# 
# This file is part of TensorArtist


from .helper import as_varnode, wrap_varnode_func
import tensorflow as tf

__all__ = [
    'concat', 'stack', 'cond_take', 'one_hot',
    'SliceOprImplHelper', 'SliceSetterOprImplHelper',
    'NormalSlice', 'NormalSliceSetter',
    'AdvancedSlice', 'AdvancedSliceSetter']


@wrap_varnode_func
def concat(inpvars, axis, name=None):
    # hack for scalar concat
    for i in inpvars:
        if as_varnode(i).ndims == 0:
            assert axis == 0
            return stack(inpvars, axis=0, name=name)
    return tf.concat(inpvars, axis=axis, name=name)


@wrap_varnode_func
def stack(inpvars, axis=0, name=None):
    return tf.stack(inpvars, axis=axis, name=name)


@wrap_varnode_func
def cond_take(tensor, mask, name=None):
    return tf.boolean_mask(tensor, mask, name=name)


@wrap_varnode_func
def one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None):
    return tf.one_hot(indices, depth, on_value=on_value, off_value=off_value, axis=axis, dtype=dtype, name=name)


class SliceOprImplHelper(object):
    def __init__(self, owner):
        self._owner = owner

    def __getitem__(self, slices):
        if type(slices) is not tuple:
            slices = (slices, )
        return self._slice_get(slices)

    def _slice_get(self, slices):
        pass


class SliceSetterOprImplHelper(SliceOprImplHelper):
    class Setter(object):
        def __init__(self, owner, slices):
            self._owner = owner
            self._slices = slices

        def __call__(self, value, *args, **kwargs):
            self._owner._slice_set(self._slices, value, *args, **kwargs)

    def _slice_get(self, slices):
        return type(self).Setter(self, slices)

    def _slice_set(self, slices, value):
        pass


class NormalSlice(SliceOprImplHelper):
    def _slice_get(self, slices):
        return as_varnode(self._owner.impl.__getitem__(slices))


class NormalSliceSetter(SliceOprImplHelper):
    def _slice_get(self, slices):
        raise NotImplementedError()


def _get_advanced_index(slices):
    for i, s in enumerate(slices):
        assert s is not None, 'does not support tf.newaxis'
        assert type(s) is not slice, 'does not support slice indexing'
    if len(slices) == 1:
        return tf.expand_dims(slices[0], -1)
    return tf.stack(slices, axis=-1)


class AdvancedSlice(SliceOprImplHelper):
    def _slice_get(self, slices):
        slices = _get_advanced_index(slices)
        return as_varnode(tf.gather_nd(self._owner, slices))


class AdvancedSliceSetter(SliceSetterOprImplHelper):
    def _slice_set(self, slices, value, use_locking=None):
        raise NotImplementedError()
        # slices = _get_advanced_index(slices)
        # return as_varnode(tf.scatter_nd_update(self._owner, slices, value, use_locking=use_locking))

