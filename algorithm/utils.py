import time
import random
import string


def generate_base_name(name):
    """
    Replace {time} from current time and random letters
    """
    now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    rand = ''.join(random.sample(string.ascii_letters, 4))
    name.replace('{time}', now + rand)
