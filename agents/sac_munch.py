import torch
import torch.nn.functional as F
from torch import distributions as pyd

from . import SACAgent


class SACMunchAgent(SACAgent):
    """SAC algorithm with Munchausen reward."""
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_clip, use_tb,
                 init_temperature, learnable_temperature,
                 m_alpha, m_tau, lo):
        super().__init__(obs_shape, action_shape, device, lr, feature_dim,
                         hidden_dim, critic_target_tau, num_expl_steps,
                         update_every_steps, stddev_clip, use_tb,
                         init_temperature, learnable_temperature)

        self.m_alpha = m_alpha
        self.m_tau = m_tau
        self.lo = lo

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            # compute Munchausen_reward
            mu_m, log_std_m = self.actor(obs)
            std = log_std_m.exp()
            dist = pyd.Normal(mu_m, std)
            log_pi_a = self.m_tau * dist.log_prob(action).mean(1).unsqueeze(1).cpu()
            assert log_pi_a.shape == (obs.shape[0], 1)
            munchausen_reward = (reward + self.m_alpha * torch.clamp(log_pi_a, min=self.lo, max=0))
            assert munchausen_reward.shape == (obs.shape[0], 1)

            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            # target_Q = reward + (discount * target_V)
            target_Q = munchausen_reward + (discount * target_V)

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
