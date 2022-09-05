from glob import glob
from pathlib import Path

root = 'models'

for model in glob(root + '/**/model', recursive=True):
    print(model)
    pth_list = []
    for pth in glob(model + '/*.pth'):
        pth_list.append(int(Path(pth).stem))

    max_pth = str(max(pth_list))

    for pth in glob(model + '/*.pth'):
        pth = Path(pth)
        if pth.stem != max_pth:
            print('remove', pth.stem)
            pth.unlink()
