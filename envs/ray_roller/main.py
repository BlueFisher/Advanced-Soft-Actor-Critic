import sys
import logging

logging.basicConfig(level=logging.INFO, format='[%(message)s')

sys.path.append('../..')
from algorithm.sac_main_hitted import MainHitted

if __name__ == '__main__':
    MainHitted(sys.argv[1:])
