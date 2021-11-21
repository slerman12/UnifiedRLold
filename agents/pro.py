# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

import utils

from blocks.augmentations import RandomShiftsAug
from blocks.encoders import Encoder
from blocks.actors import Actor, DoublePropMono
from blocks.actors import DoublePropMB
# from blocks.critics import DoubleQCritic


# todo power law, log-log, may need Q to be postive, may need extra weight
class PROAgent:
    def __init__(self, obs_shape, action_shape, discrete, device, lr, feature_dim,
                 hidden_dim, prop_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.discrete = discrete
        self.device = device
        self.prop_target_tau = prop_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        # self.min_r = 0

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        # self.prop = DoublePropMB(self.encoder.repr_dim, feature_dim, hidden_dim).to(device)
        # self.prop_target = DoublePropMB(self.encoder.repr_dim, feature_dim, hidden_dim).to(device)
        # self.prop_target.load_state_dict(self.prop.state_dict())

        self.prop = DoublePropMono(hidden_dim, hidden_dim, 3).to(device)
        self.prop_target = DoublePropMono(hidden_dim, hidden_dim, 3).to(device)
        self.prop_target.load_state_dict(self.prop.state_dict())

        # self.critic = DoubleQCritic(self.encoder.repr_dim, action_shape, feature_dim,
        #                             hidden_dim).to(device)
        # self.critic_target = DoubleQCritic(self.encoder.repr_dim, action_shape,
        #                                    feature_dim, hidden_dim).to(device)
        # self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        # self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.prop_opt = torch.optim.Adam(self.prop.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        # self.critic_target.train()
        self.prop_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.prop.train(training)
        # self.critic.train(training)

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

    def critic(self, obs, action, target=False):
        dist = self.actor(obs, 1)
        log_pi = dist.log_prob(action)
        # m1, b1, m2, b2 = self.prop_target(obs) if target else self.prop(obs)

        # Q1 = torch.exp(m1 * log_pi + torch.log(b1))  # todo b can't be negative here
        # Q2 = torch.exp(m2 * log_pi + torch.log(b2))
        # Or just:
        # Q1 = torch.exp(log_pi + torch.log(b1))
        # Q2 = torch.exp(log_pi + torch.log(b2))
        # Alternatively:
        # Q1 = torch.logit(torch.exp(log_pi)) / m1 - b1
        # Q2 = torch.logit(torch.exp(log_pi)) / m2 - b2
        # Alternatively again:
        # Q1 = torch.abs(m1) * log_pi + b1
        # Q2 = torch.abs(m2) * log_pi + b2
        # Alternatively again:
        # pi = torch.exp(log_pi)
        # Q1 = torch.abs(m1) * pi + b1
        # Q2 = torch.abs(m2) * pi + b2

        # Q1 = torch.abs(m1) * log_pi.mean(-1, keepdim=True) + b1
        # Q2 = torch.abs(m2) * log_pi.mean(-1, keepdim=True) + b2

        log_pi = log_pi.mean(-1, keepdim=True)  # todo keep dims separate
        Q1, Q2 = self.prop_target(log_pi) if target else self.prop(log_pi)
        return Q1, Q2

    def critic_target(self, obs, action):
        return self.critic(obs, action, True)

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
        self.actor_opt.zero_grad(set_to_none=True)
        self.prop_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.actor_opt.step()
        self.encoder_opt.step()
        self.prop_opt.step()

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
        obs, action, reward, discount, next_obs, _ = utils.to_torch(
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
        # utils.soft_update_params(self.critic, self.critic_target,
        #                          self.critic_target_tau)

        # update prop target
        utils.soft_update_params(self.prop, self.prop_target,
                                 self.prop_target_tau)

        return metrics
