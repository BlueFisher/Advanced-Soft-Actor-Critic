import sys

import yaml

with open(f'envs/{sys.argv[1]}/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(yaml.dump(config, default_flow_style=False))
