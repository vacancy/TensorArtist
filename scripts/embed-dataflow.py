# -*- coding:utf8 -*-
# File   : embed-dataflow.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/27/17
# 
# This file is part of TensorArtist

from tartist import qs
from tartist.qs import pprint, imshow, batch_show, stprint
from tartist.core import get_logger
from tartist.core.utils.cli import load_desc
from tartist.nn import Env

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
        print('Try: qs.*, pprint(v), imshow(img), batch_show(batch), stprint(object).')
        from IPython import embed; embed()


if __name__ == '__main__':
    main()

