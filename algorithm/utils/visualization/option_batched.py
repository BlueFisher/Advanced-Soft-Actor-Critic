from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _get_vertices(x, y):
    vertices = []
    for i in range(len(x)):
        vertices.append((x[i], 0))
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
    def __init__(self, num_options: int, model_abs_dir: Optional[Path] = None) -> None:
        self.num_options = num_options
        self.model_abs_dir = model_abs_dir

        self.fig = None
        self.idx = 0

    def add_option_termination(self,
                               option: np.ndarray | torch.Tensor,
                               termination: np.ndarray | torch.Tensor,
                               save_name: str | None = None):
        """
        Args: 
            option: [batch, ]
            termination: [batch, ]
            max_batch: The max batch size of each ray
        """
        batch_size = option.shape[0]

        option = option.detach().cpu().numpy() if isinstance(option, torch.Tensor) else option
        termination = termination.detach().cpu().numpy() if isinstance(termination, torch.Tensor) else termination

        if self.fig is None:
            self.pre_terminations = [None for _ in range(batch_size)]
            self.pre_steps = [-1 for _ in range(batch_size)]

            self.fig, self.axes = plt.subplots(nrows=batch_size, ncols=1,
                                               squeeze=False,
                                               figsize=(5, 3 * batch_size))
            self.lines = [[] for _ in range(batch_size)]
            self.polygons = [[] for _ in range(batch_size)]
            self.dots = [[] for _ in range(batch_size)]

            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

            self.axes = [ax[0] for ax in self.axes]
            for ax in self.axes:
                ax.set_autoscaley_on(True)

            for i in range(batch_size):
                for j in range(self.num_options):
                    self.lines[i].append(self.axes[i].plot([], [], '-')[0])

                self.axes[i].set_prop_cycle(None)  # Force to use the same color cycle

                for j in range(self.num_options):
                    self.polygons[i].append(self.axes[i].fill_between([], [], alpha=0.5))

                self.axes[i].set_prop_cycle(None)  # Force to use the same color cycle

                for j in range(self.num_options):
                    self.dots[i].append(self.axes[i].plot([], [], 'o')[0])

            plt.show(block=False)

            plt.pause(0.1)

        self.fig.canvas.restore_region(self._bg)
        for i in range(batch_size):
            pre_step = self.pre_steps[i]
            for j in range(self.num_options):
                hl = self.lines[i][j]
                hp = self.polygons[i][j]
                hd = self.dots[i][j]

                if option[i] == j:
                    if len(hl.get_ydata()) > 0 and np.isnan(hl.get_ydata()[-1]):
                        # Make the line continuous
                        hl.set_xdata(np.append(hl.get_xdata(), pre_step))
                        hl.set_ydata(np.append(hl.get_ydata(), self.pre_terminations[i]).astype(np.float32))

                    hl.set_xdata(np.append(hl.get_xdata(), pre_step + 1))
                    hl.set_ydata(np.append(hl.get_ydata(), termination[i]).astype(np.float32))

                    hp.set_paths(_get_vertices_list(hl.get_xdata(), hl.get_ydata()))

                    hd.set_xdata(np.append(hd.get_xdata(), pre_step + 1))
                    hd.set_ydata(np.append(hd.get_ydata(), termination[i]).astype(np.float32))
                else:
                    hl.set_xdata(np.append(hl.get_xdata(), pre_step + 0.5))  # Add a gap
                    hl.set_ydata(np.append(hl.get_ydata(), None).astype(np.float32))

                    hp.set_paths(_get_vertices_list(hl.get_xdata(), hl.get_ydata()))

                    hd.set_xdata(np.append(hd.get_xdata(), pre_step + 1))
                    hd.set_ydata(np.append(hd.get_ydata(), None).astype(np.float32))

                self.axes[i].draw_artist(self.lines[i][j])

            self.axes[i].relim()
            self.axes[i].autoscale_view()

            self.pre_terminations[i] = termination[i]
            self.pre_steps[i] = self.pre_steps[i] + 1

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.idx += 1

    def reset(self, mask):
        for i in range(len(mask)):
            if mask[i]:
                self.pre_terminations[i] = None
                self.pre_steps[i] = -1

                for j in self.num_options:
                    self.lines[i][j].set_xdata([], [])
                    self.lines[i][j].set_ydata([], [])
                    self.polygons[i][j].set_paths([])
                    self.dots[i][j].set_xdata([], [])
                    self.dots[i][j].set_ydata([], [])


if __name__ == '__main__':
    import time
    # Test OptionVisual
    option_visual = OptionVisual(num_options=4)
    for _ in range(30):
        option = torch.randint(0, 3, (2, ))
        termination = torch.rand(2)
        option_visual.add_option_termination(option, termination, max_batch=2)
        time.sleep(0.3)
    input()
