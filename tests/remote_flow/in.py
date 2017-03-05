# -*- coding:utf8 -*-
# File   : controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/2/17
# 
# This file is part of TensorArtist


from tartist.data.flow.remote.controller import control
from tartist.data.flow.remote.pipe import InputPipe
import time

q = InputPipe('tart.pipe.test')
with control(pipes=[q]):
    while True:
        print(q.get()['current'])
        time.sleep(1)

