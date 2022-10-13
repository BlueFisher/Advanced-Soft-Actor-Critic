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

        img = torch.clamp(img + torch.rand(img.shape, dtype=torch.float32, device=img.device) * self.std + self.mean, 0., 1.)

        if not is_torch:
            return T.ToPILImage()(img)
        else:
            return img


class SaltAndPepperNoise:
    """
    Args:
        snr (float): Signal Noise Rate
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

        mask = torch.rand((batch, 1, h, w), dtype=torch.float32, device=img.device).repeat(1, c, 1, 1)

        img = torch.where(mask < noise_p / 2., torch.clamp(img + self.snr, 0., 1.), img)
        img = torch.where(mask > noise_p / 2. + signal_p, torch.clamp(img - self.snr, 0., 1.), img)

        if not is_torch:
            return T.ToPILImage()(img)
        else:
            return img


class DepthNoise(object):
    """
    Args:
        p (float): Signal Noise Rate
    """

    def __init__(self, p):
        if isinstance(p, tuple):
            self.p = p
        else:
            self.p = (-p, p)

    def __call__(self, img):
        is_torch = isinstance(img, torch.Tensor)

        if not is_torch:
            img = T.ToTensor()(img)

        noise = torch.rand(1, dtype=torch.float32, device=img.device) * (self.p[1] - self.p[0]) + self.p[0]
        img = (img + noise).clip(0., 1.)

        if not is_torch:
            return T.ToPILImage()(img)
        else:
            return img


class DepthSaltAndPepperNoise(object):
    """
    Args:
        snr (float): Signal Noise Rate
    """

    def __init__(self, snr=1., p=0.03):
        self.snr = snr
        self.p = p

    def __call__(self, img):
        is_torch = isinstance(img, torch.Tensor)

        if not is_torch:
            img = T.ToTensor()(img)

        batch, c, h, w = img.shape

        noise_p = self.p
        signal_p = (1 - self.p)

        mask = torch.rand((batch, c, h, w), dtype=torch.float, device=img.device)

        img = torch.where(mask < noise_p / 2., torch.clamp(img + torch.tensor(self.snr), 0., 1.), img)
        img = torch.where(mask > noise_p / 2. + signal_p, torch.clamp(img - torch.tensor(self.snr), 0., 1.), img)

        if not is_torch:
            return T.ToPILImage()(img)
        else:
            return img