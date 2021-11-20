# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

import utils

from blocks.augmentations import RandomShiftsAug
from blocks.encoders import Encoder
from blocks.actors import Actor
from blocks.critics import DoubleQCritic
from blocks.networks import MLP


class BVSAgent:
    def __init__(self, obs_shape, action_shape, discrete, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, planner_discount, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.discrete = discrete
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.planner_discount = planner_discount
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = DoubleQCritic(self.encoder.repr_dim, action_shape, feature_dim,
                                    hidden_dim).to(device)
        self.critic_target = DoubleQCritic(self.encoder.repr_dim, action_shape,
                                           feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        action_dim = action_shape[-1]
        self.sub_planner = MLP(self.encoder.repr_dim + action_dim, 528,
                               self.encoder.repr_dim, 1).to(device)
        self.planner = MLP(self.encoder.repr_dim, 528,
                           self.encoder.repr_dim, 1).to(device)
        # TODO planner target

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.planner_opt = torch.optim.Adam(self.planner.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        # continuous vs discrete
        if self.discrete:
            return torch.argmax(action, dim=-1).cpu().numpy()[0]
        else:
            return action.cpu().numpy()[0]

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

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_planner(self, obs, action, all_obs, step, discount):
        metrics = dict()

        # for now, do 2-step only todo
        all_obs = all_obs[:, 0:2]

        all_obs = all_obs[:, 1:].float()

        obs = self.sub_planner(obs, action)

        with torch.no_grad():
            next_obs = self.aug(all_obs.view(-1, *all_obs.shape[2:]))
            next_obs = self.encoder(next_obs)

            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)

            next_obs = self.sub_planner(next_obs, next_action)

            next_obs = next_obs.view(*all_obs.shape[0:2], *next_obs.shape[1:])

            next_obs[:, -1] = self.planner(next_obs[:, -1])

            next_obs = torch.cat([obs.unsqueeze(1), next_obs], dim=1)

            discount = discount ** torch.arange(next_obs.shape[1])
            target_plan = torch.einsum('j,ijklm->iklm', discount, next_obs)

        plan = self.planner(obs)

        planner_loss = F.mse_loss(plan, target_plan)

        if self.use_tb:
            metrics['planner_loss'] = planner_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.planner_opt.zero_grad(set_to_none=True)
        planner_loss.backward()
        self.planner_opt.step()
        self.encoder_opt.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, all_obs = utils.to_torch(
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
        metrics.update(self.update_actor(obs.detach(), step))

        # update planner
        metrics.update(self.update_planner(obs, action, all_obs, step, self.planner_discount))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
