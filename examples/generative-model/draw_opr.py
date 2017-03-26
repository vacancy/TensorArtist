# -*- coding:utf8 -*-
# File   : draw_opr.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/22/17
# 
# This file is part of TensorArtist

import tensorflow as tf
from tartist.nn import opr as O


def image_diff(origin, canvas_logits):
    """
    Get the difference between original image and the canvas,
    note that the canvas is given without sigmoid activation (we will do it inside)

    :param origin: original image: batch_size, h, w, c
    :param canvas_logits: canvas logits: batch_size, h, w, c
    :return: the difference: origin - sigmoid(logits)
    """
    sigmoid_canvas = O.sigmoid(canvas_logits)
    return origin - sigmoid_canvas


def filterbank(img_h, img_w, att_dim, center_x, center_y, delta, var):
    """
    Get the filterbank matrix.

    :param img_h: image height
    :param img_w: image width
    :param att_dim: attention dim, the attention window is att_dim x att_dim
    :param center_x: attention center (x-axis): batch_size x 1
    :param center_y: attention center (y-axis): batch_size x 1
    :param delta: stride: batch_size x 1
    :param var: variance (sigma^2): batch_size x 1
    :return: filter_x, filter_y
    """

    with tf.name_scope('filterbank'):
        rng = O.range(1, att_dim+1, dtype='float32') - att_dim / 2 + 0.5
        mu_x = center_x + rng * delta
        mu_y = center_y + rng * delta
        all_x = O.range(1, img_w+1, dtype='float32')
        all_y = O.range(1, img_h+1, dtype='float32')
        a = all_x - mu_x.add_axis(-1)
        b = all_y - mu_y.add_axis(-1)

        fx = O.exp(-O.sqr(a) / var.add_axis(-1) / 2.)
        fy = O.exp(-O.sqr(b) / var.add_axis(-1) / 2.)

        fx /= fx.sum(axis=2, keepdims=True) + 1e-8
        fy /= fy.sum(axis=2, keepdims=True) + 1e-8

        fx = O.as_varnode(fx)

        return fx, fy


def split_att_params(img_h, img_w, att_dim, value):
    """
    Split attention params.

    :param img_h: image height
    :param img_w: image width
    :param att_dim: attention dim, the attention window is att_dim x att_dim
    :param value: attention params produced by hidden layer: batch_size, 5
    :return: center_x, center_y, delta, variance, gamma
    """

    with tf.name_scope('split_att_params'):
        center_x, center_y, log_delta, log_var, log_gamma = O.split(value, 5, axis=1)

        delta, var, gamma = map(lambda x: O.exp(x).reshape(-1, 1), [log_delta, log_var, log_gamma])
        center_x = ((img_w + 1.) * (center_x + 1.) / 2.).reshape(-1, 1)
        center_y = ((img_h + 1.) * (center_y + 1.) / 2.).reshape(-1, 1)
        delta *= float((max(img_h, img_w) - 1) / (att_dim - 1))
        return center_x, center_y, delta, var, gamma


def apply_filterbank(inpvar, fx, fy, dir):
    """
    Apply the filter bank, if dir is i2w, then inp var is treated as image, otherwise it is treated as window.

    :param inpvar: image: batch_size x h x w x channel
    :param fx: filter_x: batch_size x att_dim x w
    :param fy: filter_y: batch_size x att_dim x h
    :param dir: either "i2w" or "w2i" (image to window / window to image)
    :return: filtered image: batch_size, att_dim, att_dim, channel
    """

    assert dir in ('i2w', 'w2i')

    def add_channel(x, c):
        return O.tile(x, [c, 1, 1])

    with tf.name_scope('apply_filterbank'):
        h, w, c = inpvar.static_shape[1:4]
        img_h, img_w = fy.static_shape[2], fx.static_shape[2]
        att_dim = fx.static_shape[1]
        assert h is not None and w is not None and c is not None, inpvar.static_shape
        inpvar = inpvar.dimshuffle((0, 3, 1, 2)).reshape(-1, h, w)
        fx, fy = add_channel(fx, c), add_channel(fy, c)

        if dir == 'i2w':
            window = O.batch_matmul(O.batch_matmul(fy, inpvar), fx.dimshuffle(0, 2, 1))
            return window.reshape(-1, c, att_dim, att_dim).dimshuffle(0, 2, 3, 1)
        else:
            image = O.batch_matmul(O.batch_matmul(fy.dimshuffle(0, 2, 1), inpvar), fx)
            return image.reshape(-1, c, img_h, img_w).dimshuffle(0, 2, 3, 1)


def att_read(att_dim, image, center_x, center_y, delta, var):
    """
    Perform attention reading given center, delta and variance.

    :param att_dim: attention dim, the attention window is att_dim x att_dim
    :param image: image: batch_size x h x w x c
    :param center_x: attention center (x-axis): batch_size x 1
    :param center_y: attention center (y-axis): batch_size x 1
    :param delta: stride: batch_size x 1
    :param var: variance (sigma^2): batch_size x 1
    :return: attention window: batch_size x att_dim x att_dim x channel
    """

    with tf.name_scope('att_read'):
        fx, fy = filterbank(image.static_shape[1], image.static_shape[2], att_dim, center_x, center_y, delta, var)
        return apply_filterbank(image, fx, fy, dir='i2w')


def att_write(img_h, img_w, window, center_x, center_y, delta, var):
    """
    Perform attention writing given center, delta, and variance.

    :param img_h: image height
    :param img_w: image width
    :param window: window to write: batch_size x att_dim x att_dim x c
    :param center_x: attention center (x-axis): batch_size x 1
    :param center_y: attention center (y-axis): batch_size x 1
    :param delta: stride: batch_size x 1
    :param var: variance (sigma^2): batch_size x 1
    :return: a delta canvas
    """
    with tf.name_scope('att_write'):
        fx, fy = filterbank(img_h, img_w, window.static_shape[1], center_x, center_y, delta, var)
        return apply_filterbank(window, fx, fy, dir='w2i')

