import time
import logging
import getopt
import sys
sys.path.append('..')

import numpy as np

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - [%(name)s] - %(message)s')

    _log = logging.getLogger('werkzeug')
    _log.setLevel(logging.WARNING)

    _log = logging.getLogger('tensorflow')
    _log.setLevel(logging.ERROR)

    logger = logging.getLogger('sac.ds')

    node = sys.argv[1]

    if node == '-r':
        from replay import Replay
        Replay(sys.argv[2:])
    elif node == '-l':
        from hitted_main import LearnerHitted, AgentHitted

        LearnerHitted(sys.argv[2:], AgentHitted)
    elif node == '-rl':
        from hitted_main import ReplayLearnerHitted, AgentHitted

        ReplayLearnerHitted(sys.argv[2:], AgentHitted)
    elif node == '-a':
        from hitted_main import ActorHitted, AgentHitted

        ActorHitted(sys.argv[2:], AgentHitted)
    else:
        logger.error('the first arg must be one of -r, -l, -rl, or -a')
