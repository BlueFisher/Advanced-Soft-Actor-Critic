import os

import matplotlib.pyplot as plt
import numpy as np

geom = None


class VOverOptionsVisual:
    fig = None

    def __init__(self, num_options: int) -> None:
        # if 'ENABLE_OPTION_VISUAL' not in os.environ or os.environ['ENABLE_OPTION_VISUAL'] != '1':
        #     returnn

        self.num_options = num_options
        self.data = []  # Store all columns of V values

        self.fig, self.ax = plt.subplots(figsize=(8, 4))

        if geom is not None:
            self.fig.canvas.manager.window.wm_geometry(geom)

        # Initialize empty heatmap
        self.im = self.ax.imshow(np.zeros((num_options, 1)),
                                 aspect='auto',
                                 cmap='viridis',
                                 interpolation='nearest')

        self.ax.set_yticks(range(num_options))

        # Add colorbar
        self.cbar = self.fig.colorbar(self.im, ax=self.ax)
        self.cbar.set_label('V')

        plt.show(block=False)

        plt.pause(0.1)

    def add_v_over_options(self, v_over_options: np.ndarray):
        """
        Add a new column to the heatmap.

        Args:
            v_over_options: Array of shape (num_options,) containing V values for each option
        """
        if self.fig is None:
            return

        if len(v_over_options) != self.num_options:
            raise ValueError(f"Expected {self.num_options} values, got {len(v_over_options)}")

        # Add new column
        self.data.append(v_over_options)

        # Convert to 2D array: rows = options, columns = time steps
        heatmap_data = np.array(self.data).T

        # Update heatmap
        self.im.set_data(heatmap_data)
        self.im.set_extent([0, len(self.data), self.num_options, 0])

        # Auto-scale color limits
        self.im.set_clim(vmin=heatmap_data.min(), vmax=heatmap_data.max())

        # Update x-axis limits
        self.ax.set_xlim(0, len(self.data))
        self.ax.set_ylim(self.num_options, 0)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        global geom
        geom = self.fig.canvas.manager.window.geometry()

    def reset(self):
        if self.fig is None:
            return

        self.data = []
        self.im.set_data(np.zeros((self.num_options, 1)))
        self.im.set_extent([0, 1, self.num_options, 0])
        self.ax.set_xlim(0, 1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def __del__(self):
        if self.fig is None:
            return

        plt.close(self.fig)


if __name__ == '__main__':
    import time

    # Test VOverOptionsVisual
    v_visual = VOverOptionsVisual(num_options=4)
    for _ in range(20):
        # Generate random V values for each option
        v_values = np.random.randn(4)
        v_visual.add_v_over_options(v_values)
        time.sleep(0.3)
    input()
    # v_visual.reset()
    # for _ in range(30):
    #     v_values = np.random.randn(4)
    #     v_visual.add(v_values)
    #     time.sleep(0.3)
