# -*- coding:utf8 -*-
# File   : dynamic
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
# 
# This file is part of TensorArtist

import importlib
import os
import sys


__all__ = ['load_module', 'load_module_filename', 'tuple_to_classname', 'classname_to_tuple', 'load_class']


def load_module(module_name):
    module = importlib.import_module(module_name)
    return module


def load_module_filename(module_filename):
    assert module_filename.endswith('.py')
    realpath = os.path.realpath(module_filename)
    pos = realpath.rfind('/')
    dirname, module_name = realpath[0:pos], realpath[pos:-3]
    sys.path.insert(0, dirname)
    module = load_module(module_name)
    del sys.path[0]
    return module


def tuple_to_classname(t):
    assert len(t) == 2, 'Only tuple with length 2 (module name, class name) can be converted to classname, got {}, {}' \
        .format(t, len(t))

    return '.'.join(t)


def classname_to_tuple(classname):
    pos = classname.rfind('.')
    if pos == -1:
        return '', classname
    else:
        return classname[0:pos], classname[pos + 1:]


def load_class(classname, exit_on_error=True):
    if isinstance(classname, str):
        classname = classname_to_tuple(classname)
    elif isinstance(classname, tuple):
        assert len(classname) == 2, 'Classname should be tuple of length 2 or a single string'
    module_name, clz_name = classname
    try:
        module = load_module(module_name)
        clz = getattr(module, clz_name)
        return clz
    except ImportError as e:
        print('Cannot import {}.{}, with original error message: {}'.format(module_name, clz_name, str(e)))
        if exit_on_error:
            exit(1)
    return None
