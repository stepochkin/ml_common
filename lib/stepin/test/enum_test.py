# coding=utf-8

import unittest

from stepin.enum import Enum


class EnumTest(unittest.TestCase):
    def test_enum(self):
        test_enum = Enum.from_string('Test', 'en1 en2 en3 en4')
        self.assertTrue(test_enum.en1 == test_enum.en1)
        self.assertTrue(test_enum.en1 != test_enum.en2)
