#!/usr/bin/env python
# encoding: UTF8

import datetime as dtm
import logging


class _ChunkIterator(object):
    def __init__(self, it, size, first=None):
        self.it = it
        self.size = size
        self.first = first
        self.i = 0
        self.end = False

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration()
        try:
            if self.i == 0 and self.first is not None:
                res = self.first
            else:
                res = next(self.it)
            self.i += 1
            return res
        except StopIteration:
            self.end = True
            raise


class _ChunksIterator(object):
    def __init__(self, it, size):
        self.it = iter(it)
        self.size = size
        self.end = False
        self.prevCit = None

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if not self.end and self.prevCit is not None:
            self.end = self.prevCit.end
        if self.end:
            raise StopIteration()
        try:
            first = next(self.it)
        except StopIteration:
            self.end = True
            raise
        self.prevCit = _ChunkIterator(self.it, self.size, first=first)
        return self.prevCit


def chunkwise(it, size=2):
    return _ChunksIterator(it, size)


def chunkwise_str(
    t, element_to_str=None, element_separator=', ', chunk_separator='\n', size=2
):
    return chunk_separator.join(
        element_separator.join(
            str(c) if element_to_str is None else element_to_str(c) for c in cit
        )
        for cit in chunkwise(t, size)
    )


def progress_log_iter(it, log_per=1, log_interval=dtm.timedelta(minutes=1), item_count_proc=None):
    last_log_time = dtm.datetime.now()
    just_logged = False
    i = 0
    for args in it:
        yield args
        if i % log_per == 0 and dtm.datetime.now() - last_log_time > log_interval:
            logging.info('Processed %s', i)
            last_log_time = dtm.datetime.now()
            just_logged = True
        else:
            just_logged = False
        i += 1 if item_count_proc is None else item_count_proc(args)
    if not just_logged:
        logging.info('Processed %s', i)
