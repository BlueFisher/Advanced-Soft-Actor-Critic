import argparse
from glob import glob
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--episodes', action='store_true', default=False)
    args = parser.parse_args()

    root = args.root
    is_episodes = args.episodes

    if is_episodes:
        for episode in glob(root + '/**/episodes', recursive=True):
            print(episode)
            npz_list = []
            for npz in glob(episode + '/*.npz'):
                npz_list.append(int(Path(npz).stem))

            max_npz = str(max(npz_list))

            removed_npz_list = []
            for npz in glob(episode + '/*.npz'):
                npz = Path(npz)
                if npz.stem != max_npz:
                    removed_npz_list.append(npz.stem)
                    npz.unlink()
            removed_npz_list.sort()
            print('removed:', removed_npz_list.sort())

    else:
        for model in glob(root + '/**/model', recursive=True):
            print(model)
            pth_list = []
            for pth in glob(model + '/*.pth'):
                pth_list.append(int(Path(pth).stem))

            max_pth = str(max(pth_list))

            removed_pth_list = []
            for pth in glob(model + '/*.pth'):
                pth = Path(pth)
                if pth.stem != max_pth:
                    removed_pth_list.append(pth.stem)
                    pth.unlink()
            removed_pth_list.sort()
            print('removed:', removed_pth_list)
