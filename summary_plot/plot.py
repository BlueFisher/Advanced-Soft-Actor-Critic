import csv
from collections import defaultdict
from pathlib import Path
from bisect import bisect

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from cycler import cycler
from scipy import interpolate
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('seaborn-paper')
# default_cycler =  cycler(color='#e6194b,#3cb44b,#ffe119,#0082c8,#f58231,#911eb4,#46f0f0,#f032e6,#d2f53c,#fabebe,#008080,#e6beff,#aa6e28,#fffac8,#800000,#aaffc3,#808000,#ffd8b1,#000080,#808080,#ffffff,#000000'.split(','))
# plt.rc('axes', prop_cycle=default_cycler)

PRINT_STEP = 0
PRINT_ITER = 1
PRINT_TIME = 2
TIME_STAMP_LEN = 19
BASE_PATH = Path(__file__).resolve().parent.parent

def _smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)


def get_scene_paths(scene: str):
    scene_paths = list()
    for m in ['models', 'models_bak']:
        path = BASE_PATH.joinpath(m).joinpath(scene)
        scene_paths.append(path)

        print(f'{m}:')
        exp_paths_dict = get_exp_paths_dict(path)
        for exp_name in exp_paths_dict:
            if exp_name == '':
                print(f'\t\"\"')
            else:
                print(f'\t{exp_name}')

    return scene_paths


def get_exp_paths_dict(scene_path: Path) -> dict[str, list[Path]]:
    exp_paths = defaultdict(list)

    if scene_path.exists():
        for p in scene_path.iterdir():
            exp_name = p.name[:-TIME_STAMP_LEN]
            exp_paths[exp_name].append(p)

    return exp_paths


def get_fitted_xy(x, y):
    min_x = max([xx[0] for xx in x])
    max_x = min([xx[-1] for xx in x])
    max_len = 2000
    x_new = np.linspace(min_x, max_x, max_len)

    y_new = np.zeros([len(y), max_len])

    for i, xx in enumerate(x):
        yy = y[i]
        f = interpolate.interp1d(xx, yy, kind='slinear')
        y_new[i] = f(x_new)

    return x_new, y_new


# Generate csv for each tag in exp_path
def generate_cache(exp_path: Path, tags):
    log_paths: list[Path] = []

    exp_log_path = exp_path.joinpath('log')
    if exp_log_path.exists():
        log_paths.append(exp_log_path)

    for p in exp_path.iterdir():
        if p.name.startswith('learner') and p.joinpath('log').exists():
            log_paths.append(p.joinpath('log'))

    if len(log_paths) == 0:
        return False

    for i, log_path in enumerate(log_paths):
        print(f'Start generating {log_path}')

        cache = {
            tag: [[], [], []] for tag in tags
        }

        for f in log_path.iterdir():
            try:
                ea = EventAccumulator(str(f)).Reload()
                for tag in tags:
                    for event in ea.Scalars(tag):
                        cache[tag][0].append(float(event.wall_time))
                        cache[tag][1].append(int(event.step))
                        cache[tag][2].append(event.value)
            except Exception as e:
                print(e)

        for tag in cache:
            with open(exp_path.joinpath(f'{tag.replace("/","_")}_cache_{i}.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(zip(*cache[tag]))

    return True


# Get all tags of the exp with name in scene_path
def get_records(scene_path: Path, name, tags, force=False):
    if not isinstance(tags, list):
        tags = [tags]

    exp_paths = get_exp_paths_dict(scene_path)[name]

    records = {
        tag: {  # each tag has len(paths) exps
            'x_time': [],  # relative time
            'x_step': [],  # step
            'y': []
        } for tag in tags
    }
    records['_names'] = []

    if len(exp_paths) == 0:
        print(f'{name} No data')
        return

    for i, path in enumerate(exp_paths):
        gen_tags = []  # The tag of cache that needed to generate
        for tag in tags:
            if force or len(list(path.glob(f'{tag.replace("/","_")}_cache*.csv'))) == 0:
                gen_tags.append(tag)

        if len(gen_tags) > 0:
            if not generate_cache(path, gen_tags):  # Generate cache
                print(f'{path} {gen_tags} cache generating failed')

        # Read all data from cache
        for tag in tags:
            for cache_p in path.glob(f'{tag.replace("/","_")}_cache*.csv'):
                records[tag]['x_time'].append([])
                records[tag]['x_step'].append([])
                records[tag]['y'].append([])
                with open(cache_p, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for line in reader:
                        records[tag]['x_time'][-1].append(float(line[0]))
                        records[tag]['x_step'][-1].append(int(line[1]))
                        records[tag]['y'][-1].append(float(line[2]))

                # no., path, data length, the step of the last data
                print(f'{i}, {Path().joinpath(*cache_p.parts[-2:])}, iter: {len(records[tag]["x_time"][-1])}, step: {records[tag]["x_step"][-1][-1]}')

        records['_names'].append(path.name)

    for tag in tags:
        min_x_step_len = min([len(xx) for xx in records[tag]['x_step']])  # All data length is truncated to the minimum data length

        records[tag]['x_iter'] = np.arange(min_x_step_len)
        records[tag]['y_iter'] = np.array([yy[:min_x_step_len] for yy in records[tag]['y']])

        records[tag]['x_step'], records[tag]['y_step'] = get_fitted_xy(records[tag]['x_step'], records[tag]['y'])
        for i in range(len(records[tag]['x_time'])):
            records[tag]['x_time'][i] = [xx - records[tag]['x_time'][i][0] for xx in records[tag]['x_time'][i]]
        records[tag]['x_time'], records[tag]['y_time'] = get_fitted_xy(records[tag]['x_time'], records[tag]['y'])

        del records[tag]['y']

    '''
    'tag': {
        'x_time': [length, ]
        'y_time': [count, length]
        'x_step': [length, ]
        'y_step': [count, length]
        'x_iter': [length, ]
        'y_iter': [count, length]
    }
    '''
    return records


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fK' % (x * 1e-3)


formatter = ticker.EngFormatter()


def plot(exps, tags,
         print_type=PRINT_STEP,
         pdf_name=None,
         smooth=0.9,
         start_unit=None, max_unit=None, ignore=None,
         exp_options=None):
    fig, axes = plt.subplots(nrows=1, ncols=len(tags), figsize=(4 * len(tags), 3))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    if not exp_options:
        exp_options = {}

    for i, tag in enumerate(tags):
        ax = axes[i]
        for n in exps:
            r = exps[n][tag]

            if print_type == PRINT_STEP:
                x = r['x_step']
                y = r['y_step']
            elif print_type == PRINT_ITER:
                x = r['x_iter']
                y = r['y_iter']
            elif print_type == PRINT_TIME:
                x = r['x_time']
                x = x / 60 / 60
                y = r['y_time']

            if start_unit is not None:
                idx = bisect(x, start_unit)
                x = x[idx:]
                y = y[:, idx:]

            if max_unit is not None:
                idx = bisect(x, max_unit)
                x = x[:idx]
                y = y[:, :idx]

            mean_y = np.mean(y, axis=0)
            std_y = np.std(y, axis=0)

            if ignore is None:
                ignore = len(x) // 200 + 1

            max_y = mean_y + std_y
            min_y = mean_y - std_y

            x = x[:: ignore]
            mean_y = _smooth(mean_y, smooth)[:: ignore]
            max_y = _smooth(max_y, smooth)[:: ignore]
            min_y = _smooth(min_y, smooth)[:: ignore]

            base_line, = ax.plot(x, mean_y, label=n, zorder=2)
            ax.plot(x, max_y, color=base_line.get_color(), alpha=0.4, linewidth=.1, zorder=1)
            ax.plot(x, min_y, color=base_line.get_color(), alpha=0.4, linewidth=.1, zorder=1)
            ax.fill_between(x, max_y, min_y, facecolor=base_line.get_color(), alpha=0.1, zorder=1)

        if print_type == PRINT_STEP:
            ax.set_xlabel('step')
            ax.xaxis.set_major_formatter(formatter)
        elif print_type == PRINT_ITER:
            ax.set_xlabel('iteration')
            ax.xaxis.set_major_formatter(formatter)
        elif print_type == PRINT_TIME:
            ax.set_xlabel('hour')

        # ax.set_ylabel('y')
        ax.legend()
        ax.set_title(tag)

    plt.show()
    if pdf_name is not None:
        fig.savefig(f'{pdf_name}.pdf', bbox_inches='tight')


def plot_detail(exps, tag, smooth=0.9, ignore=None):
    for n in exps:
        print(n)
        r = exps[n][tag]

        x = r['x_step']
        y = r['y_step']

        fig, axes = plt.subplots(figsize=(4.5, 3))

        for i, yy in enumerate(y):
            axes.plot(x, _smooth(yy, smooth), label=exps[n]['_names'][i])

        axes.legend()
        plt.show()


if __name__ == '__main__':
    path, old_path = get_scene_paths('MountainCar')
    tags = ['reward/mean']

    exps = {
        'test': get_records(path, '', tags)
    }
