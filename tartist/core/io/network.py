# -*- coding:utf8 -*-
# File   : network.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/19/17
# 
# This file is part of TensorArtist.

from .common import fsize_format
from ..utils.cli import maybe_mkdir
from ...core.utils.thirdparty import get_tqdm_defaults
from ..logger import get_logger

import os
import hashlib
import tqdm
from six.moves import urllib

logger = get_logger(__file__)

__all__ = ['download', 'check_integrity']


def download(url, dir, filename=None, md5=None):
    """
    Download URL to a directory. Will figure out the filename automatically from URL.
    Will figure out the filename automatically from URL, if not given.
    Credit to tensorpack by Yuxin Wu
    https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/fs.py
    """
    
    maybe_mkdir(dir)
    
    filename = filename or url.split('/')[-1]
    fpath = os.path.join(dir, filename)

    def hook(t):
        last_b = [0]

        def inner(b, bsize, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return inner
    try:
        with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=filename, **get_tqdm_defaults()) as t:
            fpath, _ = urllib.request.urlretrieve(url, fpath, reporthook=hook(t))
        statinfo = os.stat(fpath)
        size = statinfo.st_size
    except:
        logger.error("Failed to download {}".format(url))
        raise
    assert size > 0, "Download an empty file!"
    logger.critical('Succesfully downloaded ' + filename + " " + fsize_format(size) + '.')

    if md5 is not None:
        assert check_integrity(fpath, md5), 'Integrity check for {} failed'.format(fpath)

    return fpath


def check_integrity(fpath, md5):
    """
    Check data integrity using md5 hashing
    Credit to torchvision
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py
    """

    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True
