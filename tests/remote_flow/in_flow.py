# -*- coding:utf8 -*-
# File   : in_flow.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/2/17
# 
# This file is part of TensorArtist.


from tartist.data import flow
import time

q = flow.RemoteFlow('tart.pipe.test')
for v in q:
    print(v)
    time.sleep(0.2)
