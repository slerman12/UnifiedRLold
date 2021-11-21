import torch
from torch import nn
from torch.nn import ParameterList

import utils

from .networks import MLP


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[-1]))

        self.apply(utils.weight_init)

    def forward(self, obs, std=0):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class DoublePropMB(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.M1 = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 1))

        self.B1 = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim, 1))

        self.M2 = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim, 1))

        self.B2 = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)

        m1 = self.M1(h)
        # b1 = torch.abs(self.B1(h)) + 0.001
        b1 = self.B1(h)
        m2 = self.M2(h)
        # b2 = torch.abs(self.B2(h)) + 0.001
        b2 = self.B2(h)
        return m1, b1, m2, b2


class DoublePropMono(nn.Module):
    def __init__(self, width, height, depth):
        super().__init__()
        self.depth = depth

        # todo val_dim
        def param():
            return ParameterList([torch.nn.Parameter(torch.Tensor(1, width, height)) for _ in range(depth)])

        self.M1 = param()
        self.B1 = param()
        self.M2 = param()
        self.B2 = param()

        self.apply(utils.weight_init)

    def forward(self, val):
        q1 = q2 = val.unsqueeze(-1)
        for l in range(self.depth):
            q1 = torch.min(torch.max(self.M1[l] * q1 + self.B1[l], dim=-1)[0], dim=-1)[0][:, None, None]
            q2 = torch.min(torch.max(self.M2[l] * q2 + self.B2[l], dim=-1)[0], dim=-1)[0][:, None, None]
        return q1.squeeze(-1), q2.squeeze(-1)


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = MLP(obs_dim, hidden_dim, 2 * action_dim, hidden_depth=hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = utils.SquashedNormal(mu, std)
        return dist




class ActorContinuous(nn.Module):
    """
    Policy network
    :param env: OpenAI gym environment
    """
    def __init__(self, env):
        super(ActorContinuous, self).__init__()
        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.shape[0]
        self.lin1 = nn.Linear(self.ds, 256)
        self.lin2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, self.da)
        self.cholesky_layer = nn.Linear(256, (self.da * (self.da + 1)) // 2)

    def forward(self, state):
        """
        forwards input through the network
        :param state: (B, ds)
        :return: mean vector (B, da) and cholesky factorization of covariance matrix (B, da, da)
        """
        device = state.device
        B = state.size(0)
        ds = self.ds
        da = self.da
        action_low = torch.from_numpy(self.env.action_space.low)[None, ...].to(device)  # (1, da)
        action_high = torch.from_numpy(self.env.action_space.high)[None, ...].to(device)  # (1, da)
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        mean = torch.sigmoid(self.mean_layer(x))  # (B, da)
        mean = action_low + (action_high - action_low) * mean
        cholesky_vector = self.cholesky_layer(x)  # (B, (da*(da+1))//2)
        cholesky_diag_index = torch.arange(da, dtype=torch.long) + 1
        cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        cholesky_vector[:, cholesky_diag_index] = F.softplus(cholesky_vector[:, cholesky_diag_index])
        tril_indices = torch.tril_indices(row=da, col=da, offset=0)
        cholesky = torch.zeros(size=(B, da, da), dtype=torch.float32).to(device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        return mean, cholesky

    def action(self, state):
        """
        :param state: (ds,)
        :return: an action
        """
        with torch.no_grad():
            mean, cholesky = self.forward(state[None, ...])
            action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
            action = action_distribution.sample()
        return action[0]


class ActorDiscrete(nn.Module):
    """
    :param env: gym environment
    """
    def __init__(self, env):
        super(ActorDiscrete, self).__init__()
        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.n
        self.lin1 = nn.Linear(self.ds, 256)
        self.lin2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, self.da)

    def forward(self, state):
        """
        :param state: (B, ds)
        :return:
        """
        B = state.size(0)
        h = F.relu(self.lin1(state))
        h = F.relu(self.lin2(h))
        h = self.out(h)
        return torch.softmax(h, dim=-1)

    def action(self, state):
        """
        :param state: (ds,)
        :return: an action
        """
        with torch.no_grad():
            p = self.forward(state[None, ...])
            action_distribution = Categorical(probs=p[0])
            action = action_distribution.sample()
        return action


