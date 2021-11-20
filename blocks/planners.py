import torch
from torch import nn
import utils
from blocks.networks import MLP


class Planner(nn.Module):
    """Critic network, employs clipped double Q learning."""
    def __init__(self, repr_dim, feature_dim, hidden_dim, output_dim, action_shape=None, sub_planner=False, discrete=False):
        super().__init__()
        self.discrete = discrete
        self.sub_planner = sub_planner

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        if not discrete and sub_planner:  # todo discrete stuff
            assert action_shape is not None
            feature_dim = feature_dim + action_shape[-1]

        self.P = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, action_shape[-1] * output_dim if discrete else output_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, action=None):
        h = self.trunk(obs)
        if not self.discrete and self.sub_planner:  # todo discrete stuff
            assert action is not None
            h = torch.cat([h, action], dim=-1)
        p = self.P(h)

        return p


class Critic(nn.Module):
    """Critic network, employs dueling Q networks."""
    def __init__(self, repr_dim, num_actions, feature_dim, hidden_dim, dueling):
        super().__init__()

        self.dueling = dueling

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        if dueling:  # todo hidden depth of 1 would suffice for Atari, can use MLP
            #  todo = MLP(feature_dim, hidden_dim, num_actions, hidden_depth)
            # todo drq paper noted two linears, not three
            self.V = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                # nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True), nn.Linear(hidden_dim, num_actions))
            self.A = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                # nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True), nn.Linear(hidden_dim, num_actions))
        else:
            self.Q = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                # nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True), nn.Linear(hidden_dim, num_actions))

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)

        if self.dueling:
            v = self.V(h)
            a = self.A(h)
            q = v + a - a.mean(1, keepdim=True)
        else:
            q = self.Q(h)

        return q
