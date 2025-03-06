import argparse
from glob import glob
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None)
    args = parser.parse_args()

    if args.root is None:
        print('root empty')
        exit()

    root = args.root

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
