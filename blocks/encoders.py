import torch
from torch import nn
import utils
from blocks.networks import Conv2d_tf


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.obs_shape = obs_shape

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.repr_dim_unflat = (32, 35, 35)

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs, flatten=True):
        # operates on last 3 dims of obs, preserves leading dims
        shape = obs.shape
        obs = obs.view(-1, *self.obs_shape)
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        if flatten:
            h = h.view(*shape[:-3], -1)
        else:
            h = h.view(*shape[:-3], *h.shape[-3:])
        return h


class SEEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape):
        super().__init__()

        self.feature_dim = 64 * 11 * 11

        self.conv1 = Conv2d_tf(obs_shape[0], 32, 8, stride=4, padding='SAME')
        self.conv2 = Conv2d_tf(32, 64, 4, stride=2, padding='SAME')
        self.conv3 = Conv2d_tf(64, 64, 3, stride=1, padding='SAME')  # todo just one for atari?

    def forward(self, obs):
        obs = obs / 255.

        h = torch.relu(self.conv1(obs))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))

        out = h.view(h.size(0), -1)
        assert out.shape[1] == self.feature_dim

        return out
