# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Uniform

import utils

from blocks.augmentations import RandomShiftsAug
from blocks.encoders import Encoder
from blocks.critics import DoubleQCritic
from . import DrQV2Agent


class DrQV2DQNAgent(DrQV2Agent):
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, discrete=True):
        super(DrQV2DQNAgent, self).__init__(obs_shape, action_shape, True, device, lr, feature_dim,
                                            hidden_dim, critic_target_tau, num_expl_steps,
                                            update_every_steps, stddev_schedule, stddev_clip, use_tb)
        self.actor = self.critic

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        Q1, Q2 = self.actor(obs)
        Q = torch.min(Q1, Q2)
        if eval_mode:
            action = torch.argmax(Q, dim=-1).cpu().numpy()[0]
        else:
            # temp = utils.schedule(self.stddev_schedule, step)  # todo decreasing temp
            temp = 1
            action = Categorical(logits=Q / temp).sample()
            if step < self.num_expl_steps:
                action = torch.randint_like(action, 0, Q.shape[-1])
        return action

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    # def update_actor(self, obs, step):
    #     metrics = dict()
    #
    #     stddev = utils.schedule(self.stddev_schedule, step)
    #     dist = self.actor(obs, stddev)
    #     action = dist.sample(clip=self.stddev_clip)
    #     log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    #     Q1, Q2 = self.critic(obs, action)
    #     Q = torch.min(Q1, Q2)
    #
    #     actor_loss = -Q.mean()
    #
    #     # optimize actor
    #     self.actor_opt.zero_grad(set_to_none=True)
    #     actor_loss.backward()
    #     self.actor_opt.step()
    #
    #     if self.use_tb:
    #         metrics['actor_loss'] = actor_loss.item()
    #         metrics['actor_logprob'] = log_prob.mean().item()
    #         metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
    #
    #     return metrics

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

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

        # update actor
        # metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
