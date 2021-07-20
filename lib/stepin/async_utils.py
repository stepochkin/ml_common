# noinspection PyProtectedMember
from pydoc import locate


class UniversalObjectFactory(object):
    def __init__(self, klass, args):
        assert klass is not None
        self.klass = klass
        self.args = args

    async def __call__(self):
        klass = locate(self.klass)
        if klass is None:
            raise Exception('Class not found for path: ' + self.klass)
        try:
            return klass(**self.args)
        except Exception as e:
            raise Exception('Error creating class %s instance with args %s' % (klass, self.args)) from e
