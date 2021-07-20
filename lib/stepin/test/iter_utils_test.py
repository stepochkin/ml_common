# encoding=UTF8

import datetime as dtm
import io
import logging.config
import os
import unittest
from time import sleep

from stepin.iter_utils import chunkwise, progress_log_iter


def mpath(*pp):
    return os.path.join(os.path.dirname(__file__), *pp)


class IterUtilsTest(unittest.TestCase):
    def test_chunkwise(self):
        data = []
        for cit in chunkwise(range(10), size=4):
            data.append(tuple(cit))
        self.assertEqual([(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)], data)

    def test_progress_log_iter(self):
        data = []
        for i in progress_log_iter(range(3)):
            data.append(i)
        self.assertEqual([0, 1, 2], data)

    def test_progress_log_iter_args(self):
        data = []
        for item in progress_log_iter(
            ((i, i+4) for i in range(3))
        ):
            data.append(item)
        self.assertEqual([(0, 4), (1, 5), (2, 6)], data)

    def test_progress_log_iter_messages(self):
        test_messages = io.StringIO()
        logging.config.dictConfig(dict(
            version=1,
            disable_existing_loggers=True,
            formatters=dict(
                concise=dict(
                    format='[%(levelname)s] %(name)s %(message)s',
                    datefmt='%H:%M:%S'
                ),
            ),
            handlers=dict(
                default={
                    'level': 'DEBUG',
                    'formatter': 'concise',
                    'class': 'logging.StreamHandler',
                    'stream': test_messages
                },
            ),
            loggers={
                '': dict(
                    handlers=['default'],
                    level='DEBUG',
                    propagate=True
                ),
            }
        ))

        data = []
        for i in progress_log_iter(range(3), log_interval=dtm.timedelta(milliseconds=1)):
            sleep(0.01)
            data.append(i)
        self.assertEqual([0, 1, 2], data)
        self.assertEqual(
            """\
[INFO] root Processed 0
[INFO] root Processed 1
[INFO] root Processed 2
""",
            test_messages.getvalue()
        )


if __name__ == '__main__':
    unittest.main()
