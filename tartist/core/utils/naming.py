# -*- coding:utf8 -*-
# File   : naming.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/23/16
# 
# This file is part of TensorArtist.
# This file is part of NeuArtist2

import os
import socket

__all__ = ['get_dump_directory', 'get_uri_prefix', 'get_name_of_vars']


def get_uri_prefix(group_name='neuart'):
    return "{}.{}.{}.".format(group_name, os.getenv('USER'), socket.gethostname())


def get_dump_directory(filename, prefix=None, suffix=''):
    if prefix is None:
        prefix = os.getenv('TART_DIR_DUMP', os.path.expanduser('~/dump'))
    dirname = os.path.basename(os.path.dirname(filename))
    dirname = dirname.replace('_', '-')

    filename = os.path.basename(filename)
    if filename.startswith('desc_'):
        filename = filename[5:]
    if filename.endswith('.py'):
        filename = filename[:-3]
    filename = filename.replace('_', '-')
    dump_dir = os.path.join(prefix, dirname, '{}{}'.format(filename, suffix))
    
    from .cli import maybe_mkdir
    return maybe_mkdir(dump_dir)


def get_data_directory(dirname, prefix=None):
    if prefix is None:
        prefix = os.getenv('TART_DIR_DATA', os.path.expanduser('~/data'))
    return os.path.join(prefix, dirname)


def get_name_of_vars(var_list):
    return [v.name for v in var_list]
