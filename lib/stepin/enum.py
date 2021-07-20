# coding=utf-8


class EnumInstance(object):
    def __init__(self, enum_name, name, code):
        self.enum_name = enum_name
        self.name = name
        self.code = code

    def __eq__(self, other):
        return self.code == other.code

    def __str__(self):
        return '<' + self.enum_name + '.' + self.name + ': ' + str(self.code) + '>'

    def __repr__(self):
        return '<' + self.enum_name + '.' + repr(self.name) + ': ' + repr(self.code) + '>'


class Enum(object):
    def __init__(self, *enums):
        self.enums = {en.name: en for en in enums}

    @staticmethod
    def from_string(enum_name, names, factory=None):
        if factory is None:
            factory = EnumInstance
        return Enum(*[factory(enum_name, name, i + 1) for i, name in enumerate(names.split(' '))])

    def __getattr__(self, name):
        en = self.enums.get(name)
        if en is None:
            raise AttributeError
        return en
