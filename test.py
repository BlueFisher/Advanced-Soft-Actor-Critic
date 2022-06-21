import logging
import unittest

logging.basicConfig(level=logging.DEBUG)

from tests.test_conv_model import *
from tests.test_nn_model import *
from tests.test_sac_main import *

unittest.main(buffer=False, verbosity=2)
