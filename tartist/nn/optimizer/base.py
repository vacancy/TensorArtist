# -*- coding:utf8 -*-
# File   : base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/28/17
# 
# This file is part of TensorArtist


import tensorflow as tf

SGDOptimizer = tf.train.GradientDescentOptimizer
MomentumOptimizer = tf.train.MomentumOptimizer
AdamOptimizer = tf.train.AdamOptimizer

AdagradOptimizer = tf.train.AdagradOptimizer
RMSPropOptimizer = tf.train.RMSPropOptimizer

