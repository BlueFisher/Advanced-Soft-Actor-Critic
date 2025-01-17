from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _get_vertices(x, y):
    vertices = []
    for i in range(len(x)):
        vertices.append((x[i], -0.2))
    for i in range(len(x) - 1, -1, -1):
        vertices.append((x[i], y[i]))
    return vertices


def _get_vertices_list(x, y):
    vertices_list = []

    x_tmp = []
    y_tmp = []
    for x_i, y_i in zip(x, y):
        if np.isnan(x_i) or np.isnan(y_i):
            if len(x_tmp) > 1:
                vertices_list.append(_get_vertices(x_tmp, y_tmp))
            x_tmp, y_tmp = [], []
        else:
            x_tmp.append(x_i)
            y_tmp.append(y_i)

    if len(x_tmp) > 1:
        vertices_list.append(_get_vertices(x_tmp, y_tmp))

    return vertices_list


class OptionVisual:
    def __init__(self, option_names: list[str]) -> None:
        self.num_options = len(option_names)

        self.pre_termination = None
        self.pre_step = -1

        self.fig, self.ax = plt.subplots(figsize=(8, 4))

        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        self.lines = [self.ax.plot([], [], '-', linewidth=1)[0] for _ in range(self.num_options)]
        self.ax.set_prop_cycle(None)  # Force to use the same color cycle
        self.polygons = [self.ax.fill_between([], [], alpha=0.5) for _ in range(self.num_options)]
        self.ax.set_prop_cycle(None)  # Force to use the same color cycle
        self.dots = [self.ax.plot([], [], 'o', markersize=1)[0] for _ in range(self.num_options)]

        self.ax.legend(option_names, loc='upper left')

        plt.show(block=False)

        plt.pause(0.1)

    def add_option_termination(self,
                               option: int,
                               termination: float):
        option = int(option)
        termination = float(termination)

        self.fig.canvas.restore_region(self._bg)

        for i in range(self.num_options):
            hl = self.lines[i]
            hp = self.polygons[i]
            hd = self.dots[i]

            if option == i:
                if len(hl.get_ydata()) > 0 and np.isnan(hl.get_ydata()[-1]):
                    # Make the line continuous
                    hl.set_xdata(np.append(hl.get_xdata(), self.pre_step))
                    hl.set_ydata(np.append(hl.get_ydata(), self.pre_termination).astype(np.float32))

                hl.set_xdata(np.append(hl.get_xdata(), self.pre_step + 1))
                hl.set_ydata(np.append(hl.get_ydata(), termination).astype(np.float32))

                hp.set_paths(_get_vertices_list(hl.get_xdata(), hl.get_ydata()))

                hd.set_xdata(np.append(hd.get_xdata(), self.pre_step + 1))
                hd.set_ydata(np.append(hd.get_ydata(), termination).astype(np.float32))
            else:
                hl.set_xdata(np.append(hl.get_xdata(), self.pre_step + 0.5))  # Add a gap
                hl.set_ydata(np.append(hl.get_ydata(), None).astype(np.float32))

                hp.set_paths(_get_vertices_list(hl.get_xdata(), hl.get_ydata()))

                hd.set_xdata(np.append(hd.get_xdata(), self.pre_step + 1))
                hd.set_ydata(np.append(hd.get_ydata(), None).astype(np.float32))

            self.ax.draw_artist(self.lines[i])

        self.ax.set_xlim((self.lines[0].get_xdata()[0], self.lines[0].get_xdata()[-1] + 1))
        self.ax.set_ylim((-0.1, 1.1))

        self.pre_termination = termination
        self.pre_step += 1

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self):
        self.pre_termination = None
        self.pre_step = -1

        for i in range(self.num_options):
            self.lines[i].set_xdata([])
            self.lines[i].set_ydata([])
            self.polygons[i].set_paths([])
            self.dots[i].set_xdata([])
            self.dots[i].set_ydata([])

    def __del__(self):
        plt.close(self.fig)


if __name__ == '__main__':
    import time
    # Test OptionVisual
    option_visual = OptionVisual(option_names=['aaa', 'bbb', 'ccc', 'ddd'])
    for _ in range(5):
        option = np.random.randint(0, 4)
        termination = np.random.rand()
        option_visual.add_option_termination(option, termination)
        time.sleep(0.3)
    input()
    option_visual.close()
    input()
    # option_visual.reset()
    # for _ in range(30):
    #     option = np.random.randint(0, 4)
    #     termination = np.random.rand()
    #     option_visual.add_option_termination(option, termination)
    #     time.sleep(0.3)
