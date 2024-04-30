import matplotlib.pyplot as plt
import numpy as np


def plot_attn_weight(attn_weight: np.ndarray):
    """
    Args:
        attn_weight: [seq_q_len, seq_k_len]
    """
    seq_q_len, seq_k_len = attn_weight.shape

    fig, ax = plt.subplots(figsize=(seq_q_len / 50, seq_k_len / 50))
    attn_weight = attn_weight / attn_weight.max(axis=1, keepdims=True)

    im = ax.imshow(attn_weight)
    ax.axis('off')

    fig.tight_layout()

    return fig


def plot_episode_option_indexes(option_indexes, option_changed_indexes, num_options):
    """
    Args:
        option_indexes: [1, ep_len]
        option_changed_indexes: [1, ep_len]
    """
    # option_changed_indexes = np.unique(option_changed_indexes)
    # option_indexes = option_indexes[:, option_changed_indexes]

    ep_len = option_indexes.shape[1]

    fig, ax = plt.subplots(figsize=(ep_len / 50, 2))

    im = ax.imshow(option_indexes / num_options)
    # ax.set_xticks(np.arange(option_indexes.shape[-1]), option_changed_indexes)
    ax.set_xticks([0, option_indexes.shape[1] - 1])
    ax.set_yticks([])

    fig.tight_layout()

    return fig
