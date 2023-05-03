import logging
import unittest

logging.basicConfig(level=logging.DEBUG)

from tests.test_sac_main import *
from tests.test_sac_params import *

unittest.main(buffer=False, verbosity=2)
