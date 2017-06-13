# -*- coding:utf8 -*-
# File   : out.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/2/17
# 
# This file is part of TensorArtist


from tartist.data.rflow import control, OutputPipe
import time
import numpy

q = OutputPipe('tart.pipe.test')
with control(pipes=[q]):
    while True:
        data = {'msg': 'hello', 'current': time.time(), 'data': numpy.zeros(shape=(128, 224, 224, 3), dtype='float32')}
        print('RFlow sending', data['current'])
        q.put(data)

