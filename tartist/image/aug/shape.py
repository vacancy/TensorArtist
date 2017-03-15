# -*- coding:utf8 -*-
# File   : shape.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/14/17
# 
# This file is part of TensorArtist

from ..imgproc import resize
from ...core.utils.shape import get_2dshape
from ... import random
import itertools

__all__ = ['random_crop', 'center_crop', 'leftup_crop',
           'random_crop_random_shape', 'random_crop_and_resize']


def _get_crop_rest(img, target_shape):
    source_shape = img.shape[:2]
    target_shape = get_2dshape(target_shape)
    rest_shape = target_shape[0] - source_shape[0], target_shape[1] - source_shape[1]
    assert rest_shape[0] >= 0 and rest_shape >= 0
    return rest_shape


def _rand_2dshape(upper_bound, lower_bound=None):
    lower_bound = lower_bound or (0, ) * len(upper_bound)
    return tuple(itertools.starmap(random.randint, zip(lower_bound, upper_bound)))


def _crop(img, start, size):
    return img[start[0]:start[0] + size[0], start[1]:start[1] + size[1]]


def random_crop(img, target_shape):
    """ random crop a image. output size is target_shape """
    rest = _get_crop_rest(img, target_shape)
    start = _rand_2dshape(rest)
    return _crop(img, start, target_shape)


def center_crop(img, target_shape):
    """ center crop """
    rest = _get_crop_rest(img, target_shape)
    start = rest[0] // 2, rest[1] // 2

    return _crop(img, start, rest)


def leftup_crop(img, target_shape):
    """ left-up crop """
    rest = _get_crop_rest(img, target_shape)
    start = 0, 0

    return _crop(img, start, rest)


def random_crop_random_shape(img, max_shape, min_shape=0):
    max_shape = get_2dshape(max_shape)
    min_shape = get_2dshape(min_shape)
    assert min_shape[0] < img.shape[0] < max_shape[0] and min_shape[1] < img.shape[1] < max_shape[1]

    tar_shape = _rand_2dshape(max_shape, lower_bound=min_shape)
    return random_crop(img, tar_shape)


def random_crop_and_resize(img, max_shape, target_shape, min_shape=0):
    target_shape = get_2dshape(target_shape)
    cropped = random_crop_random_shape(img, max_shape, min_shape=min_shape)
    return resize(cropped, target_shape)
