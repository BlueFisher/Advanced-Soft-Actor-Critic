import torch
import numpy as np


def squash_correction_log_prob(dist, x):
    return dist.log_prob(x) - torch.log(torch.maximum(1 - torch.square(torch.tanh(x)), torch.tensor(1e-2)))


def squash_correction_prob(dist, x):
    return torch.exp(dist.log_prob(x)) / (torch.maximum(1 - torch.square(torch.tanh(x)), torch.tensor(1e-2)))


def gen_pre_n_actions(n_actions, keep_last_action=False):
    if isinstance(n_actions, torch.Tensor):
        return torch.cat([
            torch.zeros_like(n_actions[:, 0:1, ...]),
            n_actions if keep_last_action else n_actions[:, :-1, ...]
        ], dim=1)
    else:
        return np.concatenate([
            np.zeros_like(n_actions[:, 0:1, ...]),
            n_actions if keep_last_action else n_actions[:, :-1, ...]
        ], axis=1)


def scale_h(x, epsilon=0.001):
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x


def scale_inverse_h(x, epsilon=0.001):
    t = 1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon)
    return torch.sign(x) * ((torch.sqrt(t) - 1) / (2 * epsilon) - 1)


def format_global_step(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    if magnitude > 0:
        num = f'{num:.1f}'
    else:
        num = str(num)

    return '%s%s' % (num, ['', 'k', 'm', 'g', 't', 'p'][magnitude])
