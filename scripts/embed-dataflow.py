# -*- coding:utf8 -*-
# File   : embed-dataflow.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/27/17
# 
# This file is part of TensorArtist.

from tartist import cao
from tartist.cao import *
from tartist.core import get_logger
import argparse

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument(dest='desc', help='The description file module')
args = parser.parse_args()


def main():
    desc = load_desc(args.desc)

    env = Env(Env.Phase.TRAIN, master_dev='/cpu:0')
    df = desc.make_dataflow_train(env)
    for data in df:
        stprint(data)
        print('Accessible variables: env, df, data.')
        print('Try: cao.*, pprint(v), imshow(img), batch_show(batch), stprint(object).')
        from IPython import embed; embed()


if __name__ == '__main__':
    main()
