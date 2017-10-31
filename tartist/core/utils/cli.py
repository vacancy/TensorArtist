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

__all__ = ['parse_devices', 'load_desc', 'yes_or_no', 'maybe_mkdir', 'parse_args']


def parse_devices(devs):
    all_gpus = []
    def rename(id_pair):
        i, d = id_pair

        d = d.lower()
        if d == 'cpu':
            return '/cpu:0'
        if d.startswith('gpu'):
            d = d[3:]
        d = str(int(d))
        all_gpus.append(d)
        return '/gpu:' + str(i)

    devs = tuple(map(rename, enumerate(devs)))
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

    quiet = os.getenv('TART_QUIET', '')
    if quiet != '':
        quiet = quiet.lower()
        assert quiet in valid, 'Invalid TART_QUIET environ: {}.'.format(quiet) 
        choice = valid[quiet]
        sys.stdout.write('TART Quiet run:\n\t{}\n\tChoice={}\n'.format(question, 'Y' if choice else 'N'))
        return choice

    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("Invalid default answer: '%s'." % default)

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
        if yes_or_no('Directory {} does not exist, do you want to create?'.format(dirname)):
            mkdir(dirname)
    return dirname


def parse_args(parser, always_clear=True):
    args, argv = parser.parse_known_args()
    if argv:
        argv = sys.argv[:1] + argv
        print('Partial parsed argv:\n\tBefore: {}\n\tAfter: {}\t'.format(sys.argv, argv))
        sys.argv = argv
    else:
        if always_clear:
            sys.argv = sys.argv[:1]
    return args
