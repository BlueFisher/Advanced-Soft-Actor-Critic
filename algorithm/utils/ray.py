from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


class RayVisual:
    def __init__(self, model_abs_dir: Optional[Path] = None) -> None:
        self.model_abs_dir = model_abs_dir

        self.fig = None
        self.idx = 0

    def __call__(self, *rays: Union[np.ndarray, torch.Tensor], max_batch=5, save_name=None):
        """
        Args:
            *rays: [batch, ray_size, C]
                ray[..., -2] = 0 if has_hit else 1
                ray[..., -1] = hit_fraction if has_hit else 1
            max_batch: The max batch size of each ray
        """
        if len(rays[0].shape) > 3:
            rays = [ray[:, -1, ...] for ray in rays]
        rays = [ray[:max_batch] for ray in rays]
        rays = [i.detach().cpu().numpy() if isinstance(i, torch.Tensor) else i for i in rays]

        batch_size = rays[0].shape[0]

        fig_size = len(rays)

        if self.fig is None:
            self.fig, self.axes = plt.subplots(nrows=max_batch, ncols=fig_size,
                                               squeeze=False,
                                               figsize=(3 * fig_size, 3 * max_batch))
            self.scs = [[] for _ in range(max_batch)]

            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

            for i in range(max_batch):
                for j, ray in enumerate(rays):
                    # ray: [batch, ray_size, C]
                    self.axes[i][j].spines['right'].set_visible(False)
                    self.axes[i][j].spines['top'].set_visible(False)
                    self.axes[i][j].spines['left'].set_position('center')
                    self.axes[i][j].spines['bottom'].set_position('center')
                    self.axes[i][j].set_xlim(-1, 1)
                    self.axes[i][j].set_ylim(-1, 1)
                    self.scs[i].append(self.axes[i][j].scatter([], [], s=1))

            plt.show(block=False)

            plt.pause(0.1)

        self.fig.canvas.restore_region(self._bg)
        for i in range(batch_size):
            for j, ray in enumerate(rays):
                mask = ray[i, :, -2] == 0.
                ray_rad = np.linspace(0, np.pi, len(ray[i]))[mask]
                ray_x = np.cos(ray_rad) * ray[i, :, -1][mask]
                ray_y = np.sin(ray_rad) * ray[i, :, -1][mask]
                self.scs[i][j].set_offsets(np.c_[ray_x, ray_y])
                self.axes[i][j].draw_artist(self.scs[i][j])

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
        self.idx += 1


if __name__ == '__main__':
    import time

    image_visual = RayVisual()
    while True:
        rays = [np.random.rand(2, 720, 2), np.random.rand(2, 720, 2)]
        image_visual(*rays)
        time.sleep(0.1)
