import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - [%(name)s] - %(message)s')

logger = logging.getLogger('test')
logger.info('aaa')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - [%(name)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info('bbb')