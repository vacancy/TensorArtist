# -*- coding:utf8 -*-
# File   : image.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/28/17
# 
# This file is part of TensorArtist

from tartist import image
import numpy as np


def imshow(img, resize=(600, 800), title='imshow'):
    """
    Image show with different parameter order.

    :param img: Image.
    :param resize: Resize factor, a tuple (min_dim, max_dim).
    :param title: The title of the shown window.
    """
    img = image.resize_minmax(img, *resize, interpolation='NEAREST')
    image.imshow(title, img)


def batch_show(batch, nr_show=16, grid_desc=('4h', '4v'), resize=(600, 800), title='batch_show'):
    """
    Show a batch of images.

    :param batch: The batched data: can be either a ndarray of shape (batch_size, h, w, c) or a list
    of images.
    :param nr_show: Number of images to be displayed. Default set to be 16.
    :param grid_desc: Grid description. See `tartist.image.image_grid` for details.
    :param resize: Resize factor, a tuple (min_dim, max_dim).
    :param title: The title of the shown window.
    """

    batch = batch[:nr_show]
    batch = np.array(batch)

    if len(batch) < 16:
        batch = np.concatenate([
            batch,
            np.zeros([16 - len(batch), batch.shape[1], batch.shape[2], batch.shape[3]], dtype=batch.dtype)
        ], axis=0)

    img = image.image_grid(batch, grid_desc)
    img = image.resize_minmax(img, *resize, interpolation='NEAREST')
    image.imshow(title, img)

