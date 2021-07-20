import asyncio
import logging
import threading
from datetime import datetime

from stepin.enum import EnumInstance, Enum


class Service(object):
    def start_async(self):
        raise NotImplementedError()

    def is_running(self):
        raise NotImplementedError()

    def state(self):
        raise NotImplementedError()

    def stop_async(self):
        raise NotImplementedError()

    def await_running(self, timeout=None):
        raise NotImplementedError()

    def await_terminated(self, timeout=None):
        raise NotImplementedError()

    def failure_cause(self):
        raise NotImplementedError()

    def add_listener(self, listener, executor):
        raise NotImplementedError()


class State(EnumInstance):
    def __init__(self, name, code):
        super(State, self).__init__('State', name, code)

    def is_terminal(self):
        raise NotImplementedError()


class TerminalState(State):
    def __init__(self, name, code):
        super(TerminalState, self).__init__(name, code)

    def is_terminal(self):
        return True


class NonTerminalState(State):
    def __init__(self, name, code):
        super(NonTerminalState, self).__init__(name, code)

    def is_terminal(self):
        return False


States = Enum(
    NonTerminalState('NEW', 1),
    NonTerminalState('STARTING', 2),
    NonTerminalState('RUNNING', 3),
    NonTerminalState('STOPPING', 4),
    TerminalState('TERMINATED', 5),
    TerminalState('FAILED', 6)
)


class Listener(object):
    def starting(self):
        pass

    def running(self):
        pass

    def stopping(self):
        pass

    def terminated(self):
        pass

    def failed(self):
        pass


class IllegalStateException(Exception):
    pass


class Callback(object):
    def __init__(self, method, method_call):
        self.method = method
        self.method_call = method_call

    def call(self, listener):
        self.method(listener)

    def enqueue_on(self, queues):
        for queue in queues:
            queue.add(self)


class ListenerCallQueue(object):
    def __init__(self, listener, executor=None):
        self.listener = listener
        self.executor = executor
        self.wait_queue = []
        self.is_thread_scheduled = False
        self.cv = threading.Condition()

    def add(self, callback):
        self.wait_queue.append(callback)

    def execute(self):
        schedule_task_runner = False
        with self.cv:
            if not self.is_thread_scheduled:
                self.is_thread_scheduled = True
                schedule_task_runner = True
        if schedule_task_runner:
            try:
                if self.executor is None:
                    self.__call__()
                else:
                    self.executor.execute(self)
            except Exception:
                # reset state in case of an error so that later calls to execute will actually do something
                with self.cv:
                    self.is_thread_scheduled = False
                # Log it and keep going.
                logging.exception(
                    "Exception while running callbacks for " + str(self.listener) +
                    " on " + str(self.executor)
                )
                raise

    def __call__(self):
        still_running = True
        try:
            while True:
                with self.cv:
                    # Preconditions.checkState(isThreadScheduled);
                    if len(self.wait_queue) > 0:
                        next_to_run = self.wait_queue.pop(0)
                    else:
                        self.is_thread_scheduled = False
                        still_running = False
                        break

                # Always run while _not_ holding the lock, to avoid deadlocks.
                try:
                    next_to_run.call(self.listener)
                except Exception:
                    # Log it and keep going.
                    logging.exception(
                        "Exception while executing callback: " + str(self.listener) +
                        "." + str(next_to_run.method_call)
                    )
        finally:
            if still_running:
                # An Error is bubbling up.We should mark ourselves as no longer running.
                # That way, if anyone tries to keep using us, we won't be corrupted.
                with self.cv:
                    self.is_thread_scheduled = False


def __abstract_service_init(cls):
    cls.STARTING_CALLBACK = Callback(lambda listener: listener.starting(), 'starting()')
    cls.RUNNING_CALLBACK = Callback(lambda listener: listener.running(), 'running()')

    def _stopping_callback(from_state):
        return Callback(
            lambda listener: listener.stopping(),
            "stopping({from = " + str(from_state) + "})"
        )

    cls.STOPPING_FROM_STARTING_CALLBACK = _stopping_callback(States.STARTING)
    cls.STOPPING_FROM_RUNNING_CALLBACK = _stopping_callback(States.RUNNING)

    def _terminated_callback(from_state):
        return Callback(
            lambda listener: listener.terminated(),
            "terminated({from = " + str(from_state) + "})"
        )

    cls.TERMINATED_FROM_STOPPING_CALLBACK = _terminated_callback(States.STOPPING)
    cls.TERMINATED_FROM_NEW_CALLBACK = _terminated_callback(States.NEW)
    cls.TERMINATED_FROM_RUNNING_CALLBACK = _terminated_callback(States.RUNNING)

    return cls


@__abstract_service_init
class AbstractService(Service):
    """
    Base class for implementing services that can handle {@link #do_start} and {@link #do_stop}
    requests, responding to them with {@link #notify_started()} and {@link #notify_stopped()}
    callbacks.
    Its subclasses must manage threads manually if you need only a single execution thread.
    """
    def __init__(self):
        self._set_state(States.NEW)
        self._listeners = []
        # noinspection PyProtectedMember
        self.cv = threading.Condition(threading._PyRLock())

    def do_start(self):
        """This method is called by {@link #start_async} to initiate service startup.
        The invocation of this method should cause a call to {@link #notify_started()},
        either during this method's run,
        or after it has returned. If startup fails, the invocation should cause a call to {@link
        #notify_failed(Exception)} instead.
        This method should return promptly; prefer to do work on a different thread where it is
        convenient. It is invoked exactly once on service startup, even when {@link #start_async} is
        called multiple times.
        """
        raise NotImplementedError()

    def do_stop(self):
        """
This method should be used to initiate service shutdown. The invocation of this method should
cause a call to {@link #notify_stopped()}, either during this method's run, or after it has
returned. If shutdown fails, the invocation should cause a call to {@link
#notify_failed(Throwable)} instead.
This method should return promptly; prefer to do work on a different thread where it is
convenient. It is invoked exactly once on service shutdown, even when {@link #stop_async} is
called multiple times.
        """
        raise NotImplementedError()

    def _is_occupied_by_current_thread(self):
        # noinspection PyProtectedMember
        return threading.get_ident() == self.cv._lock._owner

    def _set_state(self, state, shutdown_when_startup_finishes=False, failure=None):
        self._state = state
        self._shutdown_when_startup_finishes = shutdown_when_startup_finishes
        self._failure = failure

    def start_async(self):
        self.cv.acquire()
        try:
            if self._state != States.NEW:
                raise IllegalStateException(
                    'Service ' + self.__str__() + " has already been started"
                )
            self._set_state(States.STARTING)
            self._starting()
            self.do_start()
        except Exception as e:
            self.notify_failed(e)
        finally:
            self.cv.notify_all()
            self.cv.release()
            self._execute_listeners()
        return self

    def stop_async(self):
        self.cv.acquire()
        try:
            if self._state.code > States.RUNNING.code:
                return self
            previous = self._state
            if previous == States.NEW:
                self._set_state(States.TERMINATED)
                self._terminated(States.NEW)
            elif previous == States.STARTING:
                self._set_state(States.STARTING, True, None)
                self._stopping(States.STARTING)
            elif previous == States.RUNNING:
                self._set_state(States.STOPPING, False, None)
                self._stopping(States.STARTING)
                self.do_stop()
            elif previous in (States.STOPPING, States.TERMINATED, States.FAILED):
                # These cases are impossible due to the if statement above.
                raise Exception("isStoppable is incorrectly implemented, saw: " + str(previous))
            else:
                raise Exception("Unexpected state: " + str(previous))
        except Exception as e:
            self.notify_failed(e)
        finally:
            self.cv.notify_all()
            self.cv.release()
            self._execute_listeners()
        return self

    def await_running(self, timeout=None):
        self.cv.acquire()
        try:
            if timeout is None:
                until = None
            else:
                until = datetime.now() + timeout
            while self._state.code < States.RUNNING.code:
                if until is None:
                    self.cv.wait()
                else:
                    now = datetime.now()
                    if now >= until:
                        raise Exception(
                            "Timed out waiting for " + str(self) + " to reach the RUNNING state."
                        )
                    self.cv.wait((until - now).total_seconds())
            self._check_current_state(States.RUNNING)
        finally:
            self.cv.notify_all()
            self.cv.release()

    def await_terminated(self, timeout=None):
        self.cv.acquire()
        try:
            if timeout is None:
                until = None
            else:
                until = datetime.now() + timeout
            while not self._state.is_terminal():
                if until is None:
                    self.cv.wait()
                else:
                    now = datetime.now()
                    if now >= until:
                        raise Exception(
                            "Timed out waiting for " + str(self) + " to reach the TERMINATED state."
                        )
                    self.cv.wait((until - now).total_seconds())
            self._check_current_state(States.TERMINATED)
        finally:
            self.cv.notify_all()
            self.cv.release()

    def _check_current_state(self, expected_state):
        if self._state != expected_state:
            if self._state == States.FAILED:
                # Handle this specially so that we can include the failureCause, if there is one.
                raise IllegalStateException(
                    "Expected the service %s to be %s, but the service has FAILED: %s" %
                    (self, expected_state, self.failure_cause()),
                )
            raise Exception(
                "Expected the service %s to be %s, but was %s" %
                (self, expected_state, self._state)
            )

    def notify_started(self):
        self.cv.acquire()
        try:
            # We have to examine the internal state of the snapshot here to properly handle
            # the stop while starting case.
            if self._state != States.STARTING:
                failure = IllegalStateException(
                    "Cannot notifyStarted() when the service is " + str(self._state)
                )
                self.notify_failed(failure)
                raise failure
            if self._shutdown_when_startup_finishes:
                self._set_state(States.STOPPING)
                # We don't call listeners here because we already did that when we set the
                # shutdownWhenStartupFinishes flag.
                self.do_stop()
            else:
                self._set_state(States.RUNNING)
                self._running()
        finally:
            self.cv.notify_all()
            self.cv.release()
            self._execute_listeners()

    def notify_stopped(self):
        self.cv.acquire()
        try:
            # We check the internal state of the snapshot instead of state() directly so we don't allow
            # notifyStopped() to be called while STARTING, even if stop() has already been called.
            previous = self._state
            if previous != States.STOPPING and previous != States.RUNNING:
                failure = IllegalStateException(
                    "Cannot notifyStopped() when the service is " + str(previous)
                )
                self.notify_failed(failure)
                raise failure
            self._set_state(States.TERMINATED)
            self._terminated(previous)
        finally:
            self.cv.notify_all()
            self.cv.release()
            self._execute_listeners()

    def notify_failed(self, cause):
        # checkNotNull(cause)
        self.cv.acquire()
        try:
            previous = self._state
            if previous in (States.NEW, States.TERMINATED):
                raise IllegalStateException(
                    "Failed while in state:" + str(previous) + ' cause: ' + str(cause)
                )
            elif previous in (States.RUNNING, States.STARTING, States.STOPPING):
                self._set_state(States.FAILED)
                self._failed(previous, cause)
            elif previous == States.FAILED:
                # Do nothing
                pass
            else:
                raise Exception("Unexpected state: " + str(previous))
        finally:
            self.cv.notify_all()
            self.cv.release()
            self._execute_listeners()

    def is_running(self):
        self.cv.acquire()
        try:
            return self._state == States.RUNNING
        finally:
            self.cv.release()

    def state(self):
        self.cv.acquire()
        try:
            return self._state
        finally:
            self.cv.release()

    def failure_cause(self):
        self.cv.acquire()
        try:
            return self._failure
        finally:
            self.cv.release()

    def add_listener(self, listener, executor):
        # checkNotNull(listener, "listener");
        # checkNotNull(executor, "executor");
        self.cv.acquire()
        try:
            if not self._state.is_terminal():
                self._listeners.append(ListenerCallQueue(listener, executor))
        finally:
            self.cv.notify_all()
            self.cv.release()

    def __str__(self):
        return self.__class__.__name__ + ' [' + str(self._state) + ']'

    def _execute_listeners(self):
        if not self._is_occupied_by_current_thread():
            # iterate by index to avoid concurrent modification exceptions
            for i in range(len(self._listeners)):
                self._listeners[i].execute()

    def _starting(self):
        self.STARTING_CALLBACK.enqueue_on(self._listeners)

    def _running(self):
        self.RUNNING_CALLBACK.enqueue_on(self._listeners)

    def _stopping(self, from_state):
        if from_state == States.STARTING:
            self.STOPPING_FROM_STARTING_CALLBACK.enqueue_on(self._listeners)
        elif from_state == States.RUNNING:
            self.STOPPING_FROM_RUNNING_CALLBACK.enqueue_on(self._listeners)
        else:
            raise Exception()

    def _terminated(self, from_state):
        if from_state == States.NEW:
            self.TERMINATED_FROM_NEW_CALLBACK.enqueue_on(self._listeners)
        elif from_state == States.RUNNING:
            self.TERMINATED_FROM_RUNNING_CALLBACK.enqueue_on(self._listeners)
        elif from_state == States.STOPPING:
            self.TERMINATED_FROM_STOPPING_CALLBACK.enqueue_on(self._listeners)
        else:
            raise Exception()

    def _failed(self, from_state, cause_error):
        Callback(
            lambda listener: listener.failed(from_state, cause_error),
            "failed({from = " + str(from_state) + ", cause = " + str(cause_error) + "})"
        ).enqueue_on(self._listeners)


@__abstract_service_init
class AsyncAbstractService(Service):
    """
    Base class for implementing asynchronous services that can handle {@link #do_start} and {@link #do_stop}
    requests, responding to them with {@link #notify_started()} and {@link #notify_stopped()}
    callbacks.
    Its subclasses must manage threads manually if you need only a single execution thread.
    """
    def __init__(self, loop=None):
        self._set_state(States.NEW)
        self._listeners = []
        self._loop = loop
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

    async def do_start(self):
        """This method is called by {@link #start_async} to initiate service startup.
        The invocation of this method should cause a call to {@link #notify_started()},
        either during this method's run,
        or after it has returned. If startup fails, the invocation should cause a call to {@link
        #notify_failed(Exception)} instead.
        This method should return promptly; prefer to do work on a different thread where it is
        convenient. It is invoked exactly once on service startup, even when {@link #start_async} is
        called multiple times.
        """
        raise NotImplementedError()

    async def do_stop(self):
        """
This method should be used to initiate service shutdown. The invocation of this method should
cause a call to {@link #notify_stopped()}, either during this method's run, or after it has
returned. If shutdown fails, the invocation should cause a call to {@link
#notify_failed(Throwable)} instead.
This method should return promptly; prefer to do work on a different thread where it is
convenient. It is invoked exactly once on service shutdown, even when {@link #stop_async} is
called multiple times.
        """
        raise NotImplementedError()

    def _set_state(self, state, shutdown_when_startup_finishes=False, failure=None):
        self._state = state
        self._shutdown_when_startup_finishes = shutdown_when_startup_finishes
        self._failure = failure

    async def start_async(self):
        try:
            if self._state != States.NEW:
                raise IllegalStateException(
                    'Service ' + self.__str__() + " has already been started"
                )
            self._set_state(States.STARTING)
            self._starting()
            await self.do_start()
        except Exception as e:
            self.notify_failed(e)
        finally:
            self._execute_listeners()
        return self

    def stop_async(self):
        try:
            if self._state.code > States.RUNNING.code:
                return self
            previous = self._state
            if previous == States.NEW:
                self._set_state(States.TERMINATED)
                self._terminated(States.NEW)
            elif previous == States.STARTING:
                self._set_state(States.STARTING, True, None)
                self._stopping(States.STARTING)
            elif previous == States.RUNNING:
                self._set_state(States.STOPPING, False, None)
                self._stopping(States.STARTING)
                self.do_stop()
            elif previous in (States.STOPPING, States.TERMINATED, States.FAILED):
                # These cases are impossible due to the if statement above.
                raise Exception("isStoppable is incorrectly implemented, saw: " + str(previous))
            else:
                raise Exception("Unexpected state: " + str(previous))
        except Exception as e:
            self.notify_failed(e)
        finally:
            self._execute_listeners()
        return self

    def close(self):
        self.stop_async()

    class _RunningListener(Listener):
        def __init__(self, future):
            self.future = future

        def running(self):
            self.future.set_result(True)

    async def await_running(self, timeout=None):
        if self._state.code >= States.RUNNING.code:
            return
        future = asyncio.Future(loop=self._loop)
        self.add_listener(self._RunningListener(future), None)
        await asyncio.wait_for(future, timeout=timeout, loop=self._loop)
        self._check_current_state(States.RUNNING)

    class _TerminatedListener(Listener):
        def __init__(self, future):
            self.future = future

        def terminated(self):
            self.future.set_result(True)

    async def await_terminated(self, timeout=None):
        if self._state.code >= States.TERMINATED.code:
            return
        future = asyncio.Future(loop=self._loop)
        self.add_listener(self._TerminatedListener(future), None)
        await asyncio.wait_for(future, timeout=timeout, loop=self._loop)
        self._check_current_state(States.TERMINATED)

    async def wait_closed(self, timeout=None):
        await self.await_terminated(timeout=timeout)

    def _check_current_state(self, expected_state):
        if self._state != expected_state:
            if self._state == States.FAILED:
                # Handle this specially so that we can include the failureCause, if there is one.
                raise IllegalStateException(
                    "Expected the service %s to be %s, but the service has FAILED: %s" %
                    (self, expected_state, self.failure_cause()),
                )
            raise Exception(
                "Expected the service %s to be %s, but was %s" %
                (self, expected_state, self._state)
            )

    def notify_started(self):
        try:
            # We have to examine the internal state of the snapshot here to properly handle
            # the stop while starting case.
            if self._state != States.STARTING:
                failure = IllegalStateException(
                    "Cannot notifyStarted() when the service is " + str(self._state)
                )
                self.notify_failed(failure)
                raise failure
            if self._shutdown_when_startup_finishes:
                self._set_state(States.STOPPING)
                # We don't call listeners here because we already did that when we set the
                # shutdownWhenStartupFinishes flag.
                self.do_stop()
            else:
                self._set_state(States.RUNNING)
                self._running()
        finally:
            self._execute_listeners()

    def notify_stopped(self):
        try:
            # We check the internal state of the snapshot instead of state() directly so we don't allow
            # notifyStopped() to be called while STARTING, even if stop() has already been called.
            previous = self._state
            if previous != States.STOPPING and previous != States.RUNNING:
                failure = IllegalStateException(
                    "Cannot notifyStopped() when the service is " + str(previous)
                )
                self.notify_failed(failure)
                raise failure
            self._set_state(States.TERMINATED)
            self._terminated(previous)
        finally:
            self._execute_listeners()

    def notify_failed(self, cause):
        # checkNotNull(cause)
        try:
            previous = self._state
            if previous in (States.NEW, States.TERMINATED):
                raise IllegalStateException(
                    "Failed while in state:" + str(previous) + ' cause: ' + str(cause)
                )
            elif previous in (States.RUNNING, States.STARTING, States.STOPPING):
                self._set_state(States.FAILED)
                self._failed(previous, cause)
            elif previous == States.FAILED:
                # Do nothing
                pass
            else:
                raise Exception("Unexpected state: " + str(previous))
        finally:
            self._execute_listeners()

    def is_running(self):
        return self._state == States.RUNNING

    def state(self):
        return self._state

    def failure_cause(self):
        return self._failure

    def add_listener(self, listener, executor):
        # checkNotNull(listener, "listener");
        # checkNotNull(executor, "executor");
        if not self._state.is_terminal():
            self._listeners.append(ListenerCallQueue(listener, executor))

    def __str__(self):
        return self.__class__.__name__ + ' [' + str(self._state) + ']'

    def _execute_listeners(self):
        # iterate by index to avoid concurrent modification exceptions
        for i in range(len(self._listeners)):
            self._listeners[i].execute()

    def _starting(self):
        self.STARTING_CALLBACK.enqueue_on(self._listeners)

    def _running(self):
        self.RUNNING_CALLBACK.enqueue_on(self._listeners)

    def _stopping(self, from_state):
        if from_state == States.STARTING:
            self.STOPPING_FROM_STARTING_CALLBACK.enqueue_on(self._listeners)
        elif from_state == States.RUNNING:
            self.STOPPING_FROM_RUNNING_CALLBACK.enqueue_on(self._listeners)
        else:
            raise Exception()

    def _terminated(self, from_state):
        if from_state == States.NEW:
            self.TERMINATED_FROM_NEW_CALLBACK.enqueue_on(self._listeners)
        elif from_state == States.RUNNING:
            self.TERMINATED_FROM_RUNNING_CALLBACK.enqueue_on(self._listeners)
        elif from_state == States.STOPPING:
            self.TERMINATED_FROM_STOPPING_CALLBACK.enqueue_on(self._listeners)
        else:
            raise Exception()

    def _failed(self, from_state, cause_error):
        Callback(
            lambda listener: listener.failed(from_state, cause_error),
            "failed({from = " + str(from_state) + ", cause = " + str(cause_error) + "})"
        ).enqueue_on(self._listeners)
