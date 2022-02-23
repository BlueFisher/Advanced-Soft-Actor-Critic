from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_unity_to_nn_ray_index(ray_size):
    ray_index = []

    for i in reversed(range(ray_size // 2)):
        ray_index.append((i * 2 + 1) * 2)
        ray_index.append((i * 2 + 1) * 2 + 1)
    for i in range(ray_size // 2):
        ray_index.append((i * 2 + 2) * 2)
        ray_index.append((i * 2 + 2) * 2 + 1)

    return ray_index


class RayVisual:
    def __init__(self, model_abs_dir: Optional[Path] = None) -> None:
        self.model_abs_dir = model_abs_dir

        plt.ion()

        self.fig = None
        self.idx = 0

    def __call__(self, *rays: Union[np.ndarray, torch.Tensor], save_name=None):
        """
        Args:
            *rays: [Batch, ray_size, C]
                ray[..., -1] = 0 if has_hit else 1
                ray[..., -2] = hit_fraction if has_hit else 1
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
                                               figsize=(3 * fig_size, 3 * batch_size))
            self.scs = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                for j, ray in enumerate(rays):
                    # ray: [Batch, ray_size, C]
                    self.axes[i][j].spines['right'].set_visible(False)
                    self.axes[i][j].spines['top'].set_visible(False)
                    self.axes[i][j].spines['left'].set_position('center')
                    self.axes[i][j].spines['bottom'].set_position('center')
                    self.axes[i][j].set_xlim(-1, 1)
                    self.axes[i][j].set_ylim(-1, 1)
                    self.scs[i].append(self.axes[i][j].scatter([], [], s=1))

            self.fig.canvas.draw()

        for i in range(batch_size):
            for j, ray in enumerate(rays):
                mask = ray[i, :, -2] == 0.
                ray_rad = np.linspace(0, np.pi, 400)[mask]
                ray_x = np.cos(ray_rad) * ray[i, :, -1][mask]
                ray_y = np.sin(ray_rad) * ray[i, :, -1][mask]
                self.scs[i][j].set_offsets(np.c_[ray_x, ray_y])

        self.fig.canvas.flush_events()
        self.idx += 1


if __name__ == '__main__':
    import time

    image_visual = RayVisual()
    while True:
        rays = [np.random.rand(2, 720, 2), np.random.rand(2, 720, 2)]
        image_visual(*rays)
        time.sleep(0.1)
