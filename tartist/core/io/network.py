# -*- coding:utf8 -*-
# File   : network.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/19/17
# 
# This file is part of TensorArtist

from .common import fsize_format
from ..utils.cli import maybe_mkdir
from ...core.utils.thirdparty import get_tqdm_defaults
from ..logger import get_logger

import os
import tqdm
from six.moves import urllib

logger = get_logger(__file__)

__all__ = ['download']


def download(url, dir, filename=None):
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
    return fpath

