import logging
import unittest

logging.basicConfig(level=logging.DEBUG)

from tests.test_oc_params import *

unittest.main(buffer=False, verbosity=2)
