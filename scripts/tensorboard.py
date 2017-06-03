# -*- coding:utf8 -*-
# File   : tensorboard.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 5/4/17
# 
# This file is part of TensorArtist.

from tartist.core import get_env, get_logger
from tartist.core.utils.imp import load_module_filename
from tartist.data.rflow.utils import get_addr
from tartist.plugins.trainer_enhancer.summary import _tensorboard_webserver_thread

import argparse
import os
import os.path as osp

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument(dest='desc', help='The description files', nargs='+')
parser.add_argument('-p', '--port', dest='port', default=12345)
args = parser.parse_args()


def get_tensorboard_name(filename):
    base = osp.basename(filename)
    name = osp.splitext(base)[0]
    if name.startswith('desc_'):
        name = name[5:]
    return name


def main():
    log_dirs = []
    for d in args.desc:
        desc = load_module_filename(d)
        root = desc.__envs__['dir']['root']
        tb_path = osp.join(root, 'tensorboard')
        tb_name = get_tensorboard_name(d)
        log_dirs.append('{}:{}'.format(tb_name, tb_path))
        logger.info('Enable tensorboard: {}\n  dir={}'.format(tb_name, tb_path))


    logger.info('Open your tensorboard webpage at http://{}:{}'.format(get_addr(), args.port))

    commands = ['tensorboard', '--logdir', '"' + ','.join(log_dirs) + '"', '--port', str(args.port)]
    os.system(' '.join(commands))


if __name__ == '__main__':
    main()
