# -*- coding:utf8 -*-
# File   : logging.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/09/2017
#
# This file is part of TensorArtist.

import logging
import threading
import time

__all__ = ['LoggerWrapper', 'EveryNLogger', 'AtMostNLogger', 'EveryNSecondLogger']


class LoggerWrapper(object):
    def __init__(self, logger):
        self.__logger = logger
        self.__mutex = threading.Lock()

    @property
    def logger(self):
        return self.__logger

    @property
    def mutex(self):
        return self.__mutex

    def debug(self, msg, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        self.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    fatal = critical

    def _log(self, level, msg, *args, **kwargs):
        with self.mutex:
            self._log_raw(level, msg, *args, **kwargs)

    def _log_raw(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)


class EveryNLogger(LoggerWrapper):
    def __init__(self, logger, n):
        super().__init__(logger)
        self._n = n
        self._cnt = 0

    def _log_raw(self, level, msg, *args, **kwargs):
        self._cnt += 1
        if self._cnt >= self._n:
            self._cnt = 0
            super()._log_raw(level, msg, *args, **kwargs)


class AtMostNLogger(LoggerWrapper):
    def __init__(self, logger, n):
        super().__init__(logger)
        self._n = n
        self._cnt = 0

    def _log_raw(self, level, msg, *args, **kwargs):
        if self._cnt < self._n:
            super()._log_raw(level, msg, *args, **kwargs)
            self._cnt += 1


class EveryNSecondLogger(LoggerWrapper):
    def __init__(self, logger, n):
        super().__init__(logger)
        self._n = n
        self._last = 0

    def _log(self, level, msg, *args, **kwargs):
        cur = time.time()
        if cur - self._last > self._n:
            super()._log_raw(level, msg, *args, **kwargs)
            self._last = cur
