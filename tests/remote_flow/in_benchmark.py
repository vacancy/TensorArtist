# -*- coding:utf8 -*-
# File   : in_benchmark.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 5/2/17
# 
# This file is part of TensorArtist.


from tartist.data.rflow import control, InputPipe
from threading import Thread
import time
import itertools


counter = itertools.count()
current = next(counter)
prob_interval = 1

def test_thread():
    q = InputPipe('tart.pipe.test')
    with control(pipes=[q]):
        while True:
            q.get()
            next(counter)


Thread(target=test_thread, daemon=True).start()

while True:
    previous = current
    current = next(counter)
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    nr_packs = current - previous - 1
    pps = nr_packs / prob_interval
    print('RFlow benchmark: timestamp={}, pps={}.'.format(now, pps))
    time.sleep(prob_interval)
