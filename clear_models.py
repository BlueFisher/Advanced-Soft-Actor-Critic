import os
import shutil

for root, dirs, files in os.walk('models_bak'):
    for name in dirs:
        if name == 'model':
            print(os.path.join(root, name))
            shutil.rmtree(os.path.join(root, name))
