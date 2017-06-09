# -*- coding:utf8 -*-
# File   : test_image_fbaug.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/14/17
# 
# This file is part of TensorArtist.

from tartist import image
from tartist.image.aug import cblk
import sys


if __name__ == '__main__':
    a = image.imread(sys.argv[1])
    b = cblk.fbaug(a)
    image.imshow('fbaug', b)
