import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from blocks.encoders import Encoder
from replay_buffer_slow import PrioritizedReplayBuffer

from blocks.augmentations import Intensity
from blocks.critics import Critic


class DRQLAgent(object):
    """Data regularized Q-learning: Deep Q-learning."""
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, adam_eps, max_grad_norm, min_eps, num_expl_steps, num_seed_frames,
                 critic_target_tau, double_q, dueling, intensity_scale,
                 prioritized_replay_beta0, prioritized_replay_beta_steps, use_tb, discrete=True):
        self.num_actions = action_shape[-1]
        self.device = device
        self.critic_target_tau = critic_target_tau
        # self.critic_target_update_frequency = critic_target_update_frequency
        self.max_grad_norm = max_grad_norm
        self.min_eps = min_eps
        self.num_expl_steps = num_expl_steps
        self.num_seed_frames = num_seed_frames
        self.double_q = double_q
        self.use_tb = use_tb

        assert prioritized_replay_beta0 <= 1.0
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_steps = prioritized_replay_beta_steps

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.critic = Critic(self.encoder.repr_dim, self.num_actions, feature_dim,
                             hidden_dim, dueling).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, self.num_actions, feature_dim,
                                    hidden_dim, dueling).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=adam_eps)

        # data augmentation
        self.aug = Intensity(intensity_scale)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        steps_left = self.num_expl_steps + self.num_seed_frames - step
        bonus = (1.0 - self.min_eps
                 ) * steps_left / self.num_expl_steps
        bonus = np.clip(bonus, 0., 1. - self.min_eps)
        eps = self.min_eps + bonus

        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        q = self.critic(obs)
        action = q.max(dim=1)[1]

        if step < self.num_expl_steps and np.random.rand() < eps:
            action = torch.randint(low=0, high=self.num_actions, size=action.shape)

        return action.item()

    def update_critic(self, obs, action, reward, discount, next_obs, weights):
        metrics = dict()

        with torch.no_grad():
            if self.double_q:
                next_Q = self.critic(next_obs)
                next_action = next_Q.max(dim=1)[1].unsqueeze(1)
                next_target_Q = self.critic_target(next_obs)
                next_Q = next_target_Q.gather(1, next_action)
                # target_Q = reward + (not_done * discount * next_Q)
                target_Q = reward + discount * next_Q
            else:
                next_Q = self.critic_target(next_obs)
                next_Q = next_Q.max(dim=1)[0].unsqueeze(1)
                # target_Q = reward + (not_done * discount * next_Q)
                target_Q = reward + discount * next_Q

        # get current Q estimates
        Q = self.critic(obs)
        Q = Q.gather(1, action)

        td_errors = Q - target_Q
        critic_losses = F.smooth_l1_loss(Q, target_Q, reduction='none')
        if weights is not None:
            critic_losses *= weights

        critic_loss = critic_losses.mean()

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q'] = Q.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.critic.parameters(),
                                     self.max_grad_norm)
        self.critic_opt.step()

        return td_errors.squeeze(dim=1).detach().cpu().numpy(), metrics

    def update(self, replay_buffer, step):
        metrics = dict()

        prioritized_replay = type(replay_buffer) == PrioritizedReplayBuffer

        if prioritized_replay:
            fraction = min(step / self.prioritized_replay_beta_steps, 1.0)
            beta = self.prioritized_replay_beta0 + fraction * (
                    1.0 - self.prioritized_replay_beta0)
            obs, action, reward, next_obs, not_done, weights, idxs = replay_buffer.sample_multistep(
                self.batch_size, beta, self.discount, self.multistep_return)
        else:
            batch = next(replay_buffer)
            obs, action, reward, discount, next_obs = utils.to_torch(  # todo not done in replay
                batch, self.device)
            weights = None

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        td_errors, metric = self.update_critic(obs, action, reward, discount, next_obs, weights)
        metrics.update(metric)

        if prioritized_replay:
            prios = np.abs(td_errors) + 1e-6
            replay_buffer.update_priorities(idxs, prios)

        # if step % self.critic_target_update_frequency == 0:
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
