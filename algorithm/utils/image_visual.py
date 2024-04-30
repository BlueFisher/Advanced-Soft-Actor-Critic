from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


class ImageVisual:
    def __init__(self, model_abs_dir: Optional[Path] = None) -> None:
        self.model_abs_dir = model_abs_dir

        self.fig = None
        self.idx = 0

    def __call__(self, *images: np.ndarray | torch.Tensor, max_batch=5, save_name=None):
        """
        Supporting RGB and gray images
        Only supporting images with batch size less than 5
        Args:
            *images: list([batch, H, W, C], [batch, H, W, C], ...)
            max_batch: The max batch size of each image
        """
        if len(images[0].shape) > 4:
            images = [image[:, -1, ...] for image in images]
        images = [image[:max_batch] for image in images]
        images = [i.detach().cpu().numpy() if isinstance(i, torch.Tensor) else i for i in images]

        batch_size = images[0].shape[0]

        fig_size = len(images)

        if self.fig is None:
            self.fig, self.axes = plt.subplots(nrows=max_batch, ncols=fig_size,
                                               squeeze=False,
                                               figsize=(3 * fig_size, 3 * max_batch))
            self.ims = [[] for _ in range(max_batch)]

            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

            for i in range(max_batch):
                for j, image in enumerate(images):
                    self.axes[i][j].axis('off')
                    self.ims[i].append(self.axes[i][j].imshow(image[i], vmin=0, vmax=1))

            plt.show(block=False)

            plt.pause(0.1)

        self.fig.canvas.restore_region(self._bg)
        for i in range(batch_size):
            for j, image in enumerate(images):
                self.ims[i][j].set_data(image[i])
                self.axes[i][j].draw_artist(self.ims[i][j])

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

        if self.model_abs_dir:
            save_name = '' if save_name is None else save_name + '-'
            self.fig.savefig(self.model_abs_dir.joinpath(f'{save_name}{self.idx}.jpg'),
                             bbox_inches='tight',
                             pad_inches=0)
        self.idx += 1


if __name__ == '__main__':
    import time

    image_visual = ImageVisual()
    while True:
        images = [np.random.rand(2, 84, 84, 3), np.random.rand(2, 84, 84, 3)]
        image_visual(*images)
        time.sleep(0.1)
