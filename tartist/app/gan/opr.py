# -*- coding:utf8 -*-
# File   : opr.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 5/25/17
# 
# This file is part of TensorArtist


from tartist.nn import opr as O

__all__ = ['sigmoid_gan_loss']


def sigmoid_gan_loss(logits, real):
    if real:
        return O.sigmoid_cross_entropy_with_logits(logits=logits, labels=O.ones_like(logits))
    else:
        return O.sigmoid_cross_entropy_with_logits(logits=logits, labels=O.zeros_like(logits))
