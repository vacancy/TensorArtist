# -*- coding:utf8 -*-
# File   : serialization.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/1/17
# 
# This file is part of TensorArtist.


try:
    from ..logger import get_logger
    logger = get_logger(__file__)
    info = logger.warn
except Exception as e:
    info = print


try:
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()

    loads = msgpack.loads
    dumps = msgpack.dumps

except ImportError:
    import pickle

    info('Fail to import msgpack, use built-in pickle loads/dumps.')
    loads = pickle.loads
    dumps = pickle.dumps
