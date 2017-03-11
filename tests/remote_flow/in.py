# -*- coding:utf8 -*-
# File   : controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/2/17
# 
# This file is part of TensorArtist


from tartist.data.rflow import control, InputPipe
import time

q = InputPipe('tart.pipe.test')
with control(pipes=[q]):
    for i in range(10):
        print(q.get()['current'])
        time.sleep(1)
