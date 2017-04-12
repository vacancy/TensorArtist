# -*- coding:utf8 -*-
# File   : codecs.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com 
# Date   : 4/3/17
# 
# This file is part of TensorArtist

"""Image codecs utils"""


from ._backend import cv2, opencv_only
import numpy as np

@opencv_only
def jpeg_encode(img, quality=90):
    '''
    :param img: uint8 color image array
    :type img: :class:`numpy.ndarray`
    :param int quality: quality for JPEG compression
    :return: encoded image data
    '''
    return cv2.imencode('.jpg', img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tostring()


@opencv_only
def png_encode(input, compress_level=3):
    '''
    :param numpy.ndarray input: uint8 color image array
    :param int compress_level: compress level for PNG compression
    :return: encoded image data
    '''
    assert len(input.shape) == 3 and input.shape[2] in [3, 4]
    assert input.dtype == np.uint8
    assert isinstance(compress_level, int) and 0 <= compress_level <= 9
    enc = cv2.imencode('.png', input,
                       [int(cv2.IMWRITE_PNG_COMPRESSION), compress_level])
    return enc[1].tostring()


@opencv_only
def imdecode(data, *, require_chl3=True, require_alpha=False):
    """decode images in common formats (jpg, png, etc.)

    :param data: encoded image data
    :type data: :class:`bytes`
    :param require_chl3: whether to convert gray image to 3-channel BGR image
    :param require_alpha: whether to add alpha channel to BGR image

    :rtype: :class:`numpy.ndarray`
    """
    img = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img is not None, 'failed to decode'
    if img.ndim == 2 and require_chl3:
        img = img.reshape(img.shape + (1,))
    if img.shape[2] == 1 and require_chl3:
        img = np.tile(img, (1, 1, 3))
    if img.ndim == 3 and img.shape[2] == 3 and require_alpha:
        assert img.dtype == np.uint8
        img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=2)
    return img

