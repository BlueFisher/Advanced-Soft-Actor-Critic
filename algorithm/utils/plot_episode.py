import matplotlib.pyplot as plt
import numpy as np


def plot_attn_weight(attn_weight):
    """
    Args:
        attn_weight: [seq_q_len, seq_k_len]
    """
    fig, ax = plt.subplots(figsize=(1, 1))
    attn_weight = attn_weight / attn_weight.max(axis=1, keepdims=True)

    im = ax.imshow(attn_weight)
    ax.axis('off')

    return fig


def plot_episode_option_indexes(option_indexes, option_changed_indexes, num_options):
    """
    Args:
        option_indexes: [1, ep_len]
        option_changed_indexes: [1, ep_len]
    """
    # option_changed_indexes = np.unique(option_changed_indexes)
    # option_indexes = option_indexes[:, option_changed_indexes]

    fig, ax = plt.subplots(figsize=(6, 3))

    im = ax.imshow(option_indexes / num_options)
    # ax.set_xticks(np.arange(option_indexes.shape[-1]), option_changed_indexes)
    ax.set_xticks([0, option_indexes.shape[1] - 1])
    ax.set_yticks([])

    return fig
