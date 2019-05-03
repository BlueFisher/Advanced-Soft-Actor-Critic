import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import getopt
import sys
sys.path.append('..')

node = sys.argv[1]

if node == '--replay':
    from replay import Replay
    Replay(sys.argv[2:])
elif node == '--learner':
    from learner import Learner
    Learner(sys.argv[2:])
elif node == '--actor':
    from actor import Actor
    Actor(sys.argv[2:])
else:
    print('ERROR')
