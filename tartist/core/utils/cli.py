# -*- coding:utf8 -*-
# File   : cli.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/26/17
# 
# This file is part of TensorArtist.

from .imp import load_module_filename
from ..environ import load_env as load_env_

import os
import sys

__all__ = ['parse_devices', 'load_desc', 'yes_or_no', 'maybe_mkdir']


def parse_devices(devs):
    all_gpus = []
    def rename(d):
        d = d.lower()
        if d == 'cpu':
            return '/cpu:0'
        if d.startswith('gpu'):
            d = d[3:]
        d = str(int(d))
        all_gpus.append(d)
        return '/gpu:' + d
    devs = tuple(map(rename, devs))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(all_gpus)
    return devs


def load_desc(desc_filename, load_env=True):
    module = load_module_filename(desc_filename)
    if load_env:
        load_env_(module)
    return module


def yes_or_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def maybe_mkdir(dirname):
    from ..io.fs import mkdir

    if not os.path.isdir(dirname):
        if yes_or_no('dir {} does not exist, do you want to create?'.format(dirname)):
            mkdir(dirname)
    return dirname


def parse_args(parser):
    args, argv = parser.parse_known_args()
    if argv:
        print(sys.argv)
        sys.argv = argv
    return args
