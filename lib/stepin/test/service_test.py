# coding=utf-8
import asyncio
import unittest
from threading import Thread
from time import sleep

from stepin.service import AbstractService, Listener, AsyncAbstractService


class TestExecutor(object):
    @staticmethod
    def execute(runnable):
        runnable()


class TestService(AbstractService):
    def __init__(self):
        super(TestService, self).__init__()
        self.do_start_cnt = 0
        self.do_stop_cnt = 0

    def do_start(self):
        self.do_start_cnt += 1

        def thread_target():
            # print('sleep')
            sleep(0.1)
            # print('wake_up')
            self.notify_started()

        thread = Thread(target=thread_target)
        thread.start()

    def do_stop(self):
        self.do_stop_cnt += 1
        self.notify_stopped()


class TestListener(Listener):
    def __init__(self):
        self.starting_cnt = 0
        self.running_cnt = 0
        self.stopping_cnt = 0
        self.terminated_cnt = 0
        self.failed_cnt = 0

    def starting(self):
        # print('starting')
        self.starting_cnt += 1

    def running(self):
        # print('running')
        self.running_cnt += 1

    def stopping(self):
        self.stopping_cnt += 1

    def terminated(self):
        self.terminated_cnt += 1

    def failed(self):
        self.failed_cnt += 1


class ServiceTest(unittest.TestCase):
    def test_service(self):
        # print('')
        service = TestService()
        listener = TestListener()
        service.add_listener(listener, TestExecutor())
        service.start_async()
        service.await_running()
        sleep(0.1)
        self.assertEqual(service.do_start_cnt, 1)
        service.stop_async()
        service.await_terminated()
        self.assertEqual(service.do_stop_cnt, 1)
        # print('terminated')
        self.assertEqual(listener.starting_cnt, 1)
        self.assertEqual(listener.running_cnt, 1)
        self.assertEqual(listener.stopping_cnt, 1)
        self.assertEqual(listener.terminated_cnt, 1)
        self.assertEqual(listener.failed_cnt, 0)


class AsyncTestService(AsyncAbstractService):
    def __init__(self, loop, timeout):
        super().__init__(loop=loop)
        self.timeout = timeout
        self.do_start_cnt = 0
        self.do_stop_cnt = 0

    async def do_start(self):
        self.do_start_cnt += 1

        def wakeup():
            # print('wake_up')
            self.notify_started()

        # print('sleep')
        self._loop.call_later(self.timeout, wakeup)

    def do_stop(self):
        self.do_stop_cnt += 1
        self.notify_stopped()


class AsyncServiceTest(unittest.TestCase):
    def test_service_run_at_once(self):
        loop = asyncio.new_event_loop()
        service = AsyncTestService(loop=loop, timeout=0.0)
        listener = TestListener()
        service.add_listener(listener, None)
        loop.run_until_complete(service.start_async())
        sleep(0.1)
        loop.run_until_complete(service.await_running())
        self.assertEqual(service.do_start_cnt, 1)
        service.stop_async()
        loop.run_until_complete(service.await_terminated())
        self.assertEqual(service.do_stop_cnt, 1)
        self.assertEqual(listener.starting_cnt, 1)
        self.assertEqual(listener.running_cnt, 1)
        self.assertEqual(listener.stopping_cnt, 1)
        self.assertEqual(listener.terminated_cnt, 1)
        self.assertEqual(listener.failed_cnt, 0)

    def test_service_run_delayed(self):
        loop = asyncio.new_event_loop()
        service = AsyncTestService(loop=loop, timeout=0.2)
        listener = TestListener()
        service.add_listener(listener, None)
        loop.run_until_complete(service.start_async())
        sleep(0.1)
        loop.run_until_complete(service.await_running())
        self.assertEqual(service.do_start_cnt, 1)
        service.stop_async()
        loop.run_until_complete(service.await_terminated())
        self.assertEqual(service.do_stop_cnt, 1)
        self.assertEqual(listener.starting_cnt, 1)
        self.assertEqual(listener.running_cnt, 1)
        self.assertEqual(listener.stopping_cnt, 1)
        self.assertEqual(listener.terminated_cnt, 1)
        self.assertEqual(listener.failed_cnt, 0)
