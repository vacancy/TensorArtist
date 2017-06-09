# -*- coding:utf8 -*-
# File   : req.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/19/17
# 
# This file is part of TensorArtist.

from tartist.data.rflow import QueryReqPipe
import time
import sys


req = QueryReqPipe('req', conn_info=sys.argv[1:3])
with req.activate():
    out = req.query('calc', dict(a=1, b=2))
    print(out)
    time.sleep(1)
