# -*- coding:utf8 -*-
# File   : loss.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/16/17
# 
# This file is part of TensorArtist.

import tensorflow as tf

from .helper import wrap_varnode_func, wrap_force_named_op
from .helper import lazy_O as O

__all__ = [
    'grad',
    'raw_l2_loss', 'raw_smooth_l1_loss',
    'raw_cross_entropy', 'raw_cross_entropy_prob',
    'get_masked_loss', 'get_pn_balanced_loss'
]


@wrap_varnode_func
def grad(ys, xs, grad_ys=None, name='gradients'):
    single_var = type(xs) not in (tuple, list)
    if single_var:
        xs = [xs]
    outs = tf.gradients(ys, xs, grad_ys=grad_ys, name=name)
    if single_var:
        return outs[0]
    return outs


@wrap_force_named_op
def raw_l2_loss(name, pred, label):
    loss = 0.5 * O.sqr(pred - label)
    return tf.identity(loss, name='out')


@wrap_force_named_op
def raw_smooth_l1_loss(name, pred, label, sigma=1.):
    delta = O.abs(pred - label)
    mask = (delta > float(sigma)).astype('float32')
    sigma2 = float(sigma * sigma)
    loss = mask * (delta - 0.5 / sigma2) + (1. - mask) * 0.5 * delta * delta * sigma2
    return loss


@wrap_force_named_op
def raw_cross_entropy(name, pred, label, is_onehot=False):
    raise NotImplementedError()


@wrap_force_named_op
def raw_cross_entropy_prob(name, pred, label, eps=1e-4):
    loss = -label * O.log(pred + eps) - (1. - label) * O.log(1. - pred + eps)
    return tf.identity(loss, name='out')


@wrap_force_named_op
def get_masked_loss(name, loss, mask, eps=1e-3):
    loss *= mask
    loss = loss.sum() / (mask.sum() + eps)
    return tf.identity(loss, 'out')


@wrap_force_named_op
def get_pn_balanced_loss(name, loss, label, mask=None, eps=1e-3):
    neg_mask = (label < 0.5).astype('float32')
    pos_mask = (1. - neg_mask)

    if mask is not None:
        pos_mask *= mask
        neg_mask *= mask

    nr_pos_elem = pos_mask.sum()
    nr_neg_elem = neg_mask.sum()
    nr_tot_elem = nr_pos_elem + nr_neg_elem

    pos_ratio = nr_pos_elem / (nr_tot_elem + eps)
    neg_ratio = nr_neg_elem / (nr_tot_elem + eps)

    loss = (loss * pos_mask).sum() * neg_ratio + (loss * neg_mask).sum() * pos_ratio
    loss /= pos_ratio * nr_neg_elem + neg_ratio * nr_pos_elem

    return tf.identity(loss, name='out')
