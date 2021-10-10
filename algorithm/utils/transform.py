import torch
from torchvision import transforms as T


class GaussianNoise:
    def __init__(self, mean=0., std=.1):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        is_torch = isinstance(img, torch.Tensor)

        if not is_torch:
            img = T.ToTensor()(img)

        img = torch.clamp(img + torch.rand(img.shape, device=img.device) * self.std + self.mean, 0., 1.)

        if not is_torch:
            return T.ToPILImage()(img)
        else:
            return img


class SaltAndPepperNoise:
    """
    Args:
        snr （float）: Signal Noise Rate
    """

    def __init__(self, snr=.3, p=.9):
        self.snr = snr
        self.p = p

    def __call__(self, img):
        is_torch = isinstance(img, torch.Tensor)

        if not is_torch:
            img = T.ToTensor()(img)

        batch, c, h, w = img.shape

        signal_p = self.p
        noise_p = (1 - self.p)

        mask = torch.rand((batch, 1, h, w), device=img.device).repeat(1, c, 1, 1)

        img = torch.where(mask < noise_p / 2., torch.clamp(img + self.snr, 0., 1.), img)
        img = torch.where(mask > noise_p / 2. + signal_p, torch.clamp(img - self.snr, 0., 1.), img)

        if not is_torch:
            return T.ToPILImage()(img)
        else:
            return img
