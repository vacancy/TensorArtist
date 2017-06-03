# -*- coding:utf8 -*-
# File   : fs.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/19/17
# 
# This file is part of TensorArtist.


import pickle
import gzip
import enum
import os
import errno
import numpy

try:
    from ..logger import get_logger
    logger = get_logger(__file__)
    info = logger.warning
except Exception as e:
    info = print

try:
    import joblib
except ImportError:
    joblib = numpy
    info('Fail to import joblib, use built-in numpy load/dump')


from ..utils.meta import assert_instance

__all__ = ['IOMethod', 'load', 'dump', 'link', 'makedir', 'mkdir', 'make_dir', 'assert_extension', 'make_env_dir']


class IOMethod(enum.Enum):
    PICKLE = 0
    PICKLE_GZ = 1
    NUMPY = 2
    NUMPY_RAW = 3
    TEXT = 4
    BINARY = 5


def load(path, method=None, exit_on_error=False):
    if not os.path.isfile(path):
        if exit_on_error:
            info('file {} not found.'.format(path))
        return None

    if method is None:
        method = _infer_method(path)
    assert_instance(method, IOMethod)

    def load_pickle_file(fd):
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            return pickle.load(f, encoding='latin1')

    if method == IOMethod.PICKLE:
        with open(path, 'rb') as f:
            return load_pickle_file(f)
    elif method == IOMethod.PICKLE_GZ:
        with gzip.open(path, 'rb') as f:
            return load_pickle_file(f)
    elif method == IOMethod.NUMPY:
        return joblib.load(path)
    elif method == IOMethod.NUMPY_RAW:
        with open(path, 'rb') as f:
            return numpy.load(f)
    elif method == IOMethod.TEXT:
        with open(path, 'r') as f:
            return f.readlines()
    elif method == IOMethod.BINARY:
        with open(path, 'rb') as f:
            return f.read()
    else:
        raise ValueError('Unsupported loading method: {}', method)


def dump(path, content, method=None, py_prefix='', py_suffix='', text_mode='w'):
    if method is None:
        method = _infer_method(path)
    
    assert_instance(method, IOMethod)

    path_origin = path
    if method != IOMethod.TEXT or text_mode == 'w':
        path += '.tmp'

    if method == IOMethod.PICKLE:
        with open(path, 'wb') as f:
            pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif method == IOMethod.PICKLE_GZ:
        with gzip.open(path, 'wb') as f:
            pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif method == IOMethod.NUMPY:
        joblib.dump(content, path)
    elif method == IOMethod.NUMPY_RAW:
        with open(path, 'wb') as f:
            content.dump(f)
    elif method == IOMethod.TEXT:
        with open(path, text_mode) as f:
            if type(content) in (list, tuple):
                f.writelines(content)
            else:
                f.write(str(content))
    elif method == IOMethod.BINARY:
        with open(path, 'wb') as f:
            f.write(content)
    else:
        raise ValueError('Unsupported dumping method: {}', method)

    if method != IOMethod.TEXT or text_mode == 'w':
        os.rename(path, path_origin)

    return path_origin


def link(path_origin, *paths, is_relative_path=True):
    for item in paths:
        if os.path.exists(item):
            os.remove(item)
        if is_relative_path:
            src_path = os.path.relpath(path_origin, start=os.path.dirname(item))
        else:
            src_path = path_origin
        os.symlink(src_path, item)


def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise exc
    return True


mkdir = makedir
make_dir = makedir


def _infer_method(path):
    """
    Infer the loading/dumping method provided the path.
    :param path: the path to the file.
    :return: an infered IOMethod, detail can be found in the source code.
    """
    if '.' not in path:
        method = IOMethod.TEXT
    elif path.endswith('.pkl'):
        method = IOMethod.PICKLE
    elif path.endswith('.pkl.gz'):
        method = IOMethod.PICKLE_GZ
    elif path.endswith('.npy'):
        method = IOMethod.NUMPY
    elif path.endswith('.txt'):
        method = IOMethod.TEXT
    else:
        method = IOMethod.PICKLE
    return method


def assert_extension(path, extension):
    """
    Assert that a path is ends with an extension, or add it to the result path.
    :param path: the path to be checked.
    :param extension: the extension the path should ends with.
    :return: the result path, None if input is None.
    """
    if path is None:
        return path
    if not path.endswith(extension):
        return path + extension
    return path


def make_env_dir(key, path):
    from ..environ import set_env
    make_dir(path)
    set_env(key, path)
