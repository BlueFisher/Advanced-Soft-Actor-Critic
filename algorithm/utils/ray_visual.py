from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


class RayVisual:
    def __init__(self, model_abs_dir: Optional[str] = None) -> None:
        self.model_abs_dir = model_abs_dir

        plt.ion()

        self.fig = None
        self.idx = 0

    def __call__(self, *rays: Union[np.ndarray, torch.Tensor], save_name=None):
        """
        Args:
            *rays: [Batch, L, C]
        """
        if len(rays[0].shape) > 3:
            rays = [ray[:, 0, ...] for ray in rays]
        rays = [i.detach().cpu().numpy() if isinstance(i, torch.Tensor) else i for i in rays]

        batch_size = rays[0].shape[0]
        if batch_size >= 5:
            return

        fig_size = len(rays)

        if self.fig is None:
            self.fig, self.axes = plt.subplots(nrows=batch_size, ncols=fig_size,
                                               squeeze=False,
                                               figsize=(3 * fig_size, 3 * batch_size),
                                               subplot_kw={'projection': 'polar'})
            self.scs = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                for j, ray in enumerate(rays):
                    self.axes[i][j].set_theta_offset(np.pi / 2)
                    self.axes[i][j].set_rlim(0, 1)
                    self.scs[i].append(self.axes[i][j].scatter([], [], s=5))

            self.fig.canvas.draw()

        for i in range(batch_size):
            for j, ray in enumerate(rays):
                self.scs[i][j].set_offsets(np.c_[np.linspace(-np.pi, np.pi, 720), ray[i, :, -1]])

        self.fig.canvas.flush_events()
        self.idx += 1


if __name__ == '__main__':
    import time

    image_visual = RayVisual()
    while True:
        rays = [np.random.rand(2, 720, 2), np.random.rand(2, 720, 2)]
        image_visual(*rays)
        time.sleep(0.1)
