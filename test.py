import unittest

from tests.test_conv_model import *
from tests.test_envs import *
from tests.test_nn_model import *
from tests.test_rnn import *
from tests.test_sac_ds_base import *
from tests.test_vanilla import *

unittest.main(buffer=False, verbosity=2)
