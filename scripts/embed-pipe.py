# -*- coding:utf8 -*-
# File   : embed-pipe.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 5/9/17
# 
# This file is part of TensorArtist.

from tartist import cao
from tartist.cao import *
from tartist.core import get_logger
from tartist.data.rflow import control, InputPipe

import argparse

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument(dest='pipe', help='Pipe name')
args = parser.parse_args()


def main():
    q = InputPipe(args.pipe)
    with control(pipes=[q]):
        print('Accessible variables: q.')
        print('Try: cao.*, pprint(v), imshow(img), batch_show(batch), stprint(object).')
        from IPython import embed; embed()


if __name__ == '__main__':
    main()
