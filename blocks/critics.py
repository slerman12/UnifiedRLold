import torch
from torch import nn
import utils
from blocks.networks import MLP


class DoubleQCritic(nn.Module):
    """Critic network, employs clipped double Q learning."""
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


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
