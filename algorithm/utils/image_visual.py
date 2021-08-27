import matplotlib.pyplot as plt
import numpy as np


class ImageVisual:
    def __init__(self) -> None:
        plt.ion()

        self.fig = None

    def show(self, *images: np.ndarray):
        batch_size = images[0].shape[0]
        fig_size = len(images)

        if self.fig is None:
            self.fig, self.axes = plt.subplots(nrows=batch_size, ncols=fig_size,
                                               squeeze=False,
                                               figsize=(3 * fig_size, 3 * batch_size))
            self.ims = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                for j, image in enumerate(images):
                    self.axes[i][j].axis('off')
                    self.ims[i].append(self.axes[i][j].imshow(image[i]))
        else:
            for i in range(batch_size):
                for j, image in enumerate(images):
                    self.ims[i][j].set_data(image[i])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
