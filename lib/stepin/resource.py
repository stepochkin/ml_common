# encoding=utf-8
import logging
import os
from abc import abstractmethod

_verbose = False


def set_verbose(value):
    global _verbose
    _verbose = value


class Resource(object):
    def __init__(self, dependencies=None, force=False):
        self.dependencies = list([] if dependencies is None else dependencies)
        self.force = force
        self.name = None
        self._need_create = None

    def set_name(self, value):
        self.name = value
        return self

    def set_force(self, value):
        self.force = value
        return self

    def add_dependency(self, dependency):
        self.dependencies.append(dependency)
        return self

    @abstractmethod
    def is_available(self):
        raise Exception('Not implemented')

    # do not override
    def need_create(self):
        self._need_create = any(r.need_create() for r in self.dependencies) or self.force
        return self._need_create

    # do not override
    def create_recurse(self, force):
        if force or self._need_create or not self.is_available():
            dep_data = [r.create_recurse(force) for r in self.dependencies]
            if _verbose:
                logging.debug(
                    'Creating resource %s',
                    self.__class__.__name__ if self.name is None else self.name
                )
            return self.do_create(dep_data)
        if _verbose:
            logging.debug(
                'Opening resource %s',
                self.__class__.__name__ if self.name is None else self.name
            )
        return self.open()

    # do not override
    def create(self, force=False):
        self.need_create()
        return self.create_recurse(force)

    def do_create(self, dep_data):
        pass

    def open(self):
        pass

    # do not override
    def destroy(self):
        self.do_destroy()
        for r in self.dependencies:
            r.destroy()

    def do_destroy(self):
        pass


class FileResource(Resource):
    def __init__(self, paths, dependencies=None):
        super(FileResource, self).__init__(dependencies=dependencies)
        self.paths = paths

    def is_available(self):
        return all(os.path.exists(p) for p in self.paths)
