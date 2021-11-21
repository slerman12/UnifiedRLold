# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

import utils

from . import DrQV2Agent


class rQdiaAgent(DrQV2Agent):
    def update_rQdia(self, obs, obs_orig, action):
        metrics = dict()

        # rQdia (Regularizing Q-Value Distributions With Image Augmentation)

        batch_size = action.shape[0]  # |B|

        scaling = 1  # ∈ (0, 1], lower = more efficient
        num_actions = max(1, round(batch_size * scaling))  # m

        obs_dim = obs.shape[1]
        action_dim = action.shape[1]

        obs_orig_pairs = obs_orig.unsqueeze(1).expand(-1, num_actions, -1).reshape(-1, obs_dim)  # s^(i)
        obs_pairs = obs.unsqueeze(1).expand(-1, num_actions, -1).reshape(obs_orig_pairs.shape)  # aug(s^(i))
        action_pairs = action[:num_actions].unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, action_dim)  # a^(j)

        # Q dists
        obs_orig_Q1_dist, obs_orig_Q2_dist = self.critic(obs_orig_pairs, action_pairs)
        obs_Q1_dist, obs_Q2_dist = self.critic(obs_pairs, action_pairs)

        rQdia_loss = F.mse_loss(obs_orig_Q1_dist, obs_Q1_dist) + F.mse_loss(obs_orig_Q2_dist, obs_Q2_dist)

        if self.use_tb:
            metrics['rQdia_loss'] = rQdia_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        rQdia_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()  # todo should encoder be updated too?

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(
            batch, self.device)

        # for rQdia
        obs_orig = self.encoder(obs)

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
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update rQdia
        metrics.update(
            self.update_rQdia(obs, obs_orig, action))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
