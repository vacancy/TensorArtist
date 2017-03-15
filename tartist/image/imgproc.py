# -*- coding:utf8 -*-
# File   : imgproc.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com 
# Date   : 01/22/17
# 
# This file is part of TensorArtist

from . import _backend
from ..core.utils.shape import get_2dshape
import os
import enum
import math

import numpy as np

__all__ = [
    'imread', 'imwrite', 'imshow',
    'resize', 'resize_wh',
    'resize_scale', 'resize_scale_wh',
    'crop'
]


class ShuffleType(enum.Enum):
    BC01 = 0
    B01C = 1


def imread(path, *, shuffle=False):
    if not os.path.exists(path):
        return None
    i = _backend.imread(path)
    if i is None:
        return None
    if shuffle:
        return dimshuffle(i, ShuffleType.BC01)
    return i


def imwrite(path, img, *, shuffle=False):
    if shuffle:
        img = dimshuffle(img, ShuffleType.B01C)
    _backend.imwrite(path, img)


def imshow(title, img, *, shuffle=False):
    if shuffle:
        img = dimshuffle(img, ShuffleType.B01C)
    _backend.imshow(title, img)


def resize(img, size):
    size = get_2dshape(size)
    return _backend.resize(img, (size[1], size[0]))


def resize_wh(img, size_wh):
    size_wh = get_2dshape(size_wh)
    return _backend.resize(img, size_wh)


def resize_scale(img, scale):
    scale = get_2dshape(scale, type=float)
    new_size = int(img.shape[0] * scale[0]), int(img.shape[1] * scale[1])
    return resize(img, new_size)


def resize_scale_wh(img, scale_wh):
    scale_wh = get_2dshape(scale_wh, type=float)
    return resize_scale(img, (scale_wh[1], scale_wh[0]))


def crop(image, l, t, w, h, extra_crop=None):
    if extra_crop is not None and extra_crop != 1:
        new_w, new_h = round(w * extra_crop), round(h * extra_crop)
        l -= (new_w - w) // 2
        t -= (new_h - h) // 2
        w, h = new_w, new_h

    im_h, im_w = image.shape[0:2]
    w, h = int(round(w)), int(round(h))
    l, t = int(math.floor(l)), int(math.floor(t))
    # range is expected to be image[t:t+h, l:l+w] now.

    ex_l, ex_t, ex_w, ex_h = l, t, w, h
    delta_l, delta_t = 0, 0
    if ex_l < 0:
        ex_l = 0
        delta_l = ex_l - l
        ex_w -= delta_l
    if ex_t < 0:
        ex_t = 0
        delta_t = ex_t - t
        ex_h -= delta_t
    if ex_l + ex_w > im_w:
        ex_w = im_w - ex_l
    if ex_t + ex_h > im_h:
        ex_h = im_h - ex_t

    result = np.zeros(shape=(h, w) + image.shape[2:], dtype=image.dtype)
    result[delta_t:delta_t+ex_h, delta_l:delta_l+ex_w] = image[ex_t:ex_t+ex_h, ex_l:ex_l+ex_w]
    return result


def dimshuffle(img, shuffle_type):
    assert len(img.shape) in (2, 3, 4), 'Image should be of dims 2, 3 or 4'
    assert isinstance(shuffle_type, ShuffleType)

    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        if shuffle_type == ShuffleType.BC01:
            return np.transpose(img, (2, 0, 1))
        else:
            return np.transpose(img, (1, 2, 0))
    else:  # len(img.shape) == 4:
        if shuffle_type == ShuffleType.BC01:
            return np.transpose(img, (0, 3, 1, 2))
        else:
            return np.transpose(img, (0, 2, 3, 1))

