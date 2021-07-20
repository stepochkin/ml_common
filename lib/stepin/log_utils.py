import gzip
import os
import shutil
import sys
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

import time


def log_exc(level, m=None, *args, **kwargs):
    if m is None:
        ei = sys.exc_info()
        # m = '%s: %s' % (ei[1].__class__.__name__, ei[1])
        m = str(ei[1])
    kwargs['exc_info'] = 1
    logging.log(level, m, *args, **kwargs)


def debug_exc(m=None, *args, **kwargs):
    log_exc(logging.DEBUG, m, *args, **kwargs)


def warn_exc(m=None, *args, **kwargs):
    log_exc(logging.WARNING, m, *args, **kwargs)


def error_exc(m=None, *args, **kwargs):
    log_exc(logging.ERROR, m, *args, **kwargs)


def safe_call(call, m=None, *args, **kwargs):
    try:
        call()
    except:
        warn_exc(m, *args, **kwargs)


async def safe_async_call(call, m=None, *args, **kwargs):
    try:
        await call()
    except:
        warn_exc(m, *args, **kwargs)


def safe_close(obj, m=None, *args, **kwargs):
    if obj is None:
        return
    safe_call(obj.close, m, *args, **kwargs)


async def safe_wait_closed(obj, m=None, *args, **kwargs):
    if obj is None:
        return
    await safe_async_call(obj.wait_closed, m, *args, **kwargs)


async def safe_async_named_close(obj, name):
    safe_close(obj, 'Error closing "%s"', name)
    await safe_wait_closed(obj, 'Error waiting to close "%s"', name)


async def safe_async_typed_close(obj, type_name, name):
    safe_close(obj, 'Error closing %s "%s"', type_name, name)
    await safe_wait_closed(obj, 'Error waiting to close %s "%s"', type_name, name)


def safe_stop(obj, m=None, *args, **kwargs):
    if obj is None:
        return
    safe_call(obj.stop, m=m, *args, **kwargs)


async def safe_wait_stopped(obj, m=None, *args, **kwargs):
    if obj is None:
        return
    await safe_async_call(obj.wait_stopped, m=m, *args, **kwargs)


class GzipRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.namer = self._namer
        self.rotator = self._rotator

    @staticmethod
    def _namer(name):
        return name + ".gz"

    @staticmethod
    def _rotator(source, dest):
        with open(source, 'rb') as f_in:
            with gzip.open(dest, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)


class GzipTimedRotatingFileHandler(TimedRotatingFileHandler):
    def _open(self):
        return gzip.open(self.baseFilename, self.mode + 't', encoding=self.encoding)


def _hour_time(t):
    dt = datetime.fromtimestamp(t)
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return time.mktime(dt.timetuple()) + dt.microsecond / 1E6


class ExactHourRotatingFileHandler(TimedRotatingFileHandler):
    # noinspection PyPep8Naming
    def __init__(
        self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False, atTime=None,
        isGzipped=False
    ):
        self.isGzipped = isGzipped
        super().__init__(filename, when, interval, backupCount, encoding, delay, utc, atTime)
        self.rolloverAt = self.computeRollover(time.time())

    def computeRollover(self, current_time):
        return super().computeRollover(_hour_time(current_time))

    def _open(self):
        if self.isGzipped:
            return gzip.open(self.baseFilename, self.mode + 't', encoding=self.encoding)
        return super()._open()
