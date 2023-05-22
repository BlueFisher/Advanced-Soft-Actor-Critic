import matplotlib.pyplot as plt


def plot_attn_weight(attn_weight):
    """
    Args:
        attn_weight: [query_length, key_length]
    """
    fig, ax = plt.subplots(figsize=(1, 1))
    attn_weight = attn_weight / attn_weight.max(axis=1, keepdims=True)

    im = ax.imshow(attn_weight)
    ax.axis('off')

    return fig
