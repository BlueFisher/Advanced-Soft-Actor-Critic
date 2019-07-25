import sys
import logging

logging.basicConfig(level=logging.INFO, format='[%(message)s')

sys.path.append('../..')
from algorithm.sac_main_hitted import MainHitted

if __name__ == '__main__':
    logger = logging.getLogger('sac')
    MainHitted(sys.argv[1:])

    # for i in range(4):
    #     arg = sys.argv[1:] + [f'--sac=sac{i+1}', f'--name={i+1}_{{time}}']
    #     MainHitted(arg)