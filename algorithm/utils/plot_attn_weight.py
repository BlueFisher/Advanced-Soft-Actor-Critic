import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor


def plot_attn_weight(attn_weight):
    """
    Args:
        attn_weight: [query_length, key_length]
    """
    fig, ax = plt.subplots(figsize=(2, 2))
    attn_weight /= attn_weight.max(axis=1, keepdims=True)

    im = ax.imshow(attn_weight)
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)

    return image
