# -*- coding:utf8 -*-
# File   : rep.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/19/17
# 
# This file is part of TensorArtist.

from tartist.data.rflow import QueryRepPipe
import time


def answer(pipe, identifier, inp):
    out = inp['a'] + inp['b']
    pipe.send(identifier, dict(out=out))

rep = QueryRepPipe('rep')
rep.dispatcher.register('calc', answer)
with rep.activate():
    print('tart req.py', *rep.conn_info)
    while True:
        time.sleep(1)
