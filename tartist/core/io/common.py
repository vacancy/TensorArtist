# -*- coding:utf8 -*-
# File   : common.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/19/17
# 
# This file is part of TensorArtist.

from math import log

__all__ = ['fsize_format']


unit_list = list(zip(['bytes', 'kB', 'MB', 'GB', 'TB', 'PB'], [0, 0, 1, 2, 2, 2]))
def fsize_format(num):
    """human readable file size
    from http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size"""

    if num == 0:
        return '0 bytes'
    if num == 1:
        return '1 byte'

    exponent = min(int(log(num, 1024)), len(unit_list) - 1)
    quotient = float(num) / 1024**exponent
    unit, num_decimals = unit_list[exponent]
    format_string = '{:.%sf} {}' % (num_decimals)
    return format_string.format(quotient, unit)
