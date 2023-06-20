import logging
import sys
import unittest

logging.basicConfig(level=logging.DEBUG)

from tests.test_oc_params import *

argv = []

for arg in sys.argv:
    if not arg.startswith('--from'):
        argv.append(arg)

sys.argv = argv

unittest.main(buffer=False, verbosity=2)
