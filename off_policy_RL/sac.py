import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import random
import time

from collections import deque


import gymnasium as gym

env = gym.make("InvertedPendulum-v5", render_mode='rgb_array')
env.reset()

if torch.xpu.is_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

class actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(64, action_dim)
        self.log_std_head = nn.Linear(64, action_dim)

        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)

        self.register_buffer('action_scale', (high - low) / 2)
        self.register_buffer('action_bias', (high + low) / 2)
        
    def forward(self, x):
        x = self.net(x)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2)
        return mu, log_std

    def compute_distribution(self, x):
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)
        distribution = Normal(mu, std)
        return distribution

    def compute_log_prob(self, distribution, z):
        log_prob = distribution.log_prob(z)
        log_tanh_jacobian = 2 * (math.log(2) - z - F.softplus(-2*z))
        log_prob = log_prob - log_tanh_jacobian - torch.log(self.action_scale)
        log_prob = log_prob.sum(dim=-1)
        return log_prob
        
    def sample(self, x):
        distribution = self.compute_distribution(x)
        z = distribution.rsample()
        normalized_action = torch.tanh(z)
        action = self.action_bias + self.action_scale * normalized_action
        log_prob = self.compute_log_prob(distribution, z)
        return z, action, log_prob

    def sample_repeat(self, x, k):
        distribution = self.compute_distribution(x)
        z = distribution.rsample((k,))
        normalized_action = torch.tanh(z[0])
        action = self.action_bias + self.action_scale * normalized_action
        log_prob_repeat = self.compute_log_prob(distribution, z)
        log_prob = log_prob_repeat.mean(dim=0)
        return action, log_prob

    def get_gaussian_log_prob(self, x, z, eps=1e-6):
        distribution = self.compute_distribution(x)
        return distribution.log_prob(z).sum(dim=-1)
        

class critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return  self.net(x).squeeze(-1)

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, alpha, epsilon, max_priority=1.0, n_step_return=1):
        self.max_size = max_size
        self.last_pos = 0
        self.size = 0

        self.states = np.empty((self.max_size, state_dim), dtype=np.float32)
        self.actions = np.empty((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.empty(self.max_size, dtype=np.float32)
        self.next_states = np.empty((self.max_size, state_dim), dtype=np.float32)
        self.terminated = np.empty(self.max_size, dtype=np.bool_)
        self.priorities = np.empty(self.max_size, dtype=np.float64)
        self.gamma_pow = np.empty(self.max_size, dtype=np.int32)
        self.intermediate_state  = np.empty((self.max_size, n_step_return - 1, state_dim), dtype=np.float32)
        self.intermediate_behavior_log_prob = np.empty((self.max_size, n_step_return - 1), dtype=np.float32)
        self.intermediate_action = np.empty((self.max_size, n_step_return - 1, action_dim), dtype=np.float32)
        self.intermediate_z = np.empty((self.max_size, n_step_return - 1, action_dim), dtype=np.float32)
        self.intermediate_n_return = np.empty((self.max_size, n_step_return - 1), dtype=np.float32)
        self.max_priority = max_priority
        self.alpha = alpha
        self.epsilon = epsilon
        
    def append(self, data):
        state, action, reward, next_state, terminated, gamma_pow, \
        intermediate_state, intermediate_action, intermediate_behavior_log_prob, \
        intermediate_z, intermediate_n_return = data

        self.states[self.last_pos] = np.asarray(state, dtype=np.float32)
        self.actions[self.last_pos] = action
        self.rewards[self.last_pos] = reward
        self.next_states[self.last_pos] = next_state
        self.terminated[self.last_pos] = terminated
        self.priorities[self.last_pos] = self.max_priority
        self.gamma_pow[self.last_pos] = gamma_pow
        self.intermediate_state[self.last_pos] = intermediate_state
        self.intermediate_action[self.last_pos] = intermediate_action
        self.intermediate_z[self.last_pos] = intermediate_z
        self.intermediate_behavior_log_prob[self.last_pos] = intermediate_behavior_log_prob
        self.intermediate_n_return[self.last_pos] = intermediate_n_return
        
        self.last_pos = (self.last_pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def change_priority(self, batch_indx, batch_prior):
        priorities = np.asarray(batch_prior, dtype=np.float64) + self.epsilon
        self.priorities[batch_indx] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))
    
    def sample(self, batch_size):
        scaled_priorities = self.priorities[:self.size] ** self.alpha
        sum_priority = scaled_priorities.sum()
        
        probabilities = scaled_priorities / sum_priority

        indx = np.random.choice(self.size, size=batch_size, p=probabilities)
        return (
            indx,
            torch.from_numpy(self.states[indx]),
            torch.from_numpy(self.actions[indx]),
            torch.from_numpy(self.rewards[indx]),
            torch.from_numpy(self.next_states[indx]),
            torch.from_numpy(self.terminated[indx]),
            torch.from_numpy(self.gamma_pow[indx]),
            torch.from_numpy(self.intermediate_state[indx]),
            torch.from_numpy(self.intermediate_action[indx]),
            torch.from_numpy(self.intermediate_behavior_log_prob[indx]),
            torch.from_numpy(self.intermediate_z[indx]),
            torch.from_numpy(self.intermediate_n_return[indx]),
            torch.as_tensor(probabilities[indx], dtype=torch.float32),
        )

    def __len__(self):
        return self.size
        
class SAC:
    def __init__(self, env, episodes=10000, batch_size=256, device='cpu', 
                gamma=0.99, lr=3*1e-4, buffer_size_for_start_train=2000, 
                max_size_buffer=40000, alpha_buffer=0.5, epsilon_buffer=1e-4, beta_per_weight=0.5, beta_per_weight_decay=1.05,
                n_step_return=1, tau_update_t_nets=0.005, policy_delay=2, alpha_explore=0.2):
        
        self.n_step_return = n_step_return 
        self.n_step_buffer = deque()

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        self.device = device

        self.Q_Actor = actor(self.state_dim, self.action_dim, 
                             self.env.action_space.low, self.env.action_space.high).to(self.device)
        self.Q1_Critic = critic(self.state_dim, self.action_dim).to(self.device)
        self.Q2_Critic = critic(self.state_dim, self.action_dim).to(self.device)
        
        self.T1_Critic = critic(self.state_dim, self.action_dim).to(self.device)
        self.T2_Critic = critic(self.state_dim, self.action_dim).to(self.device)
        
        self.T1_Critic.load_state_dict(self.Q1_Critic.state_dict())
        self.T2_Critic.load_state_dict(self.Q2_Critic.state_dict())
        
        self.T1_Critic.requires_grad_(False)
        self.T2_Critic.requires_grad_(False)
        self.T1_Critic.eval()
        self.T2_Critic.eval()

        self.alpha_explore = alpha_explore
        
        self.policy_delay = policy_delay
        self.tau_update_t_nets = tau_update_t_nets
        
        self.gamma = gamma
        self.loss_fn = nn.MSELoss(reduction='none')
        self.actor_optimizer = torch.optim.Adam(self.Q_Actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(list(self.Q1_Critic.parameters()) + 
                                                 list(self.Q2_Critic.parameters()), lr=lr)

        self.replay_buffer = ReplayBuffer(max_size=max_size_buffer, state_dim=self.state_dim, action_dim=self.action_dim,
                                          alpha=alpha_buffer, epsilon=epsilon_buffer, n_step_return=self.n_step_return)
        self.beta_per_weight = beta_per_weight
        self.beta_per_weight_decay = beta_per_weight_decay
        self.buffer_size_for_start_train = buffer_size_for_start_train
        
        self.episodes = episodes
        self.batch_size = batch_size
        self.update_steps = 0
        self.env_steps = 0

        self.obs_count = 0
        self.obs_mean = np.zeros(self.state_dim, dtype=np.float64)
        self.obs_m2 = np.zeros(self.state_dim, dtype=np.float64)

        self.history = {'return': [], 'critic_loss': [], 'actor_loss': [], 'success_rate': [], 
                        'episode_success': [], 'episode_length': [], 'td_abs': [], 'entropy': [], 
                        'eval_visual': dict(), 'eval_summary': []}

    def update_obs_metrics(self, observation):
        x = np.asarray(observation, dtype=np.float64)
        self.obs_count += 1

        delta = x - self.obs_mean
        self.obs_mean += delta / self.obs_count

        delta_after_update = x - self.obs_mean
        self.obs_m2 += delta * delta_after_update
        
    def preprocess(self, obs):
        if self.obs_count < 2:
            return obs
            
        mean = torch.as_tensor(self.obs_mean, dtype=obs.dtype, device=obs.device)
        variance = torch.as_tensor(self.obs_m2 / self.obs_count, dtype=obs.dtype, device=obs.device)
    
        return (obs - mean) / torch.sqrt(variance + 1e-8)

    @torch.no_grad()
    def select_action(self, state, noise_std=0.1, deterministic=False):
        state = state.to(self.device)
        # normalize_state = self.preprocess(state)
        normalize_state = state
        if deterministic:
            mu, _ = self.Q_Actor(normalize_state)
            action = (self.Q_Actor.action_bias + self.Q_Actor.action_scale * torch.tanh(mu))
            behavior_log_prob = z = None
        else:
            z, action, _ = self.Q_Actor.sample(normalize_state)
            behavior_log_prob = self.Q_Actor.get_gaussian_log_prob(state, z).item()
            z = z.cpu().numpy().astype(np.float32)
        return action.cpu().numpy().astype(np.float32), behavior_log_prob, z

    @torch.no_grad()
    def n_step_sliding_window(self):
        n_return = gamma_pow = 0
        first_state, first_action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        intermediate_action = np.zeros((self.n_step_return - 1, self.action_dim), dtype=np.float32)
        intermediate_behavior_log_prob = np.zeros(self.n_step_return - 1, dtype=np.float32)
        intermediate_z = np.zeros((self.n_step_return - 1, self.action_dim), dtype=np.float32)
        intermediate_state = np.zeros((self.n_step_return - 1, self.state_dim), dtype=np.float32)
        intermediate_n_return = np.zeros(self.n_step_return - 1, dtype=np.float32)
                                         
        for i in range(self.n_step_return):
            state, action, reward, next_state, terminated, truncated, behavior_log_prob, z = self.n_step_buffer[i]

            if i > 0:
                intermediate_action[i-1] = np.asarray(action, dtype=np.float32)
                intermediate_behavior_log_prob[i-1] = np.asarray(behavior_log_prob, dtype=np.float32)
                intermediate_state[i-1] = np.asarray(state, dtype=np.float32)
                intermediate_n_return[i-1] = np.asarray(n_return, dtype=np.float32)
                intermediate_z[i-1] = np.asarray(z, dtype=np.float32)
                
            n_return += (self.gamma ** gamma_pow) * reward
            gamma_pow += 1

            if truncated or terminated:
                break

        self.replay_buffer.append([first_state, first_action, n_return, next_state, terminated, gamma_pow, 
                                   intermediate_state, intermediate_action, intermediate_behavior_log_prob,
                                   intermediate_z, intermediate_n_return])
        self.n_step_buffer.popleft()

    def collect_data(self, state):        
        action, behavior_log_prob, z = self.select_action(state)
            
        final_reward = -100
        total_reward = 0

        next_observation, reward, terminated, truncated, _ = self.env.step(action)
        self.update_obs_metrics(next_observation)
        
        next_state = torch.tensor(next_observation, dtype=torch.float32)

        self.n_step_buffer.append([state, action, reward, next_state, terminated, truncated, behavior_log_prob, z])

        if terminated or truncated:
            while self.n_step_buffer:
                self.n_step_sliding_window()
        elif len(self.n_step_buffer) == self.n_step_return:
            self.n_step_sliding_window()
        
        state = next_state

        total_reward += reward
        if terminated:
            final_reward = reward

        return next_state, terminated, truncated, final_reward, total_reward

    @torch.no_grad()
    def update_t_nets(self):
        for online_net, target_net in ((self.Q1_Critic, self.T1_Critic), 
                                      (self.Q2_Critic, self.T2_Critic)):
            for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
                target_param.mul_(1 - self.tau_update_t_nets)
                target_param.add_(online_param, alpha=self.tau_update_t_nets)
                
    @torch.no_grad()
    def compute_target(self, next_state, n_step_return, terminated, gamma_pow,
                       intermediate_state, intermediate_n_return):

        batch_size = intermediate_state.shape[0]
        num_intermediate = intermediate_state.shape[1]

        intermediate_valid = torch.zeros(num_intermediate, batch_size, dtype=torch.bool, device=self.device)
        intermediate_target = torch.zeros(num_intermediate, batch_size, dtype=torch.float32, device=self.device)
        
        for i in range(self.n_step_return-1):
            step = i + 1
            valid = gamma_pow > step
            
            if not valid.any():
                continue

            intermediate_valid[i] = valid
            soft_i_return = intermediate_n_return[valid, i].clone()

            for j in range(i):
                state = intermediate_state[valid, j, :]
                _, logprob = self.Q_Actor.sample_repeat(state, step)
                soft_i_return -= self.gamma ** (j + 1) * self.alpha_explore * logprob

            inter_next_state = intermediate_state[valid, i]
            inter_next_action, inter_next_logprob = self.Q_Actor.sample_repeat(inter_next_state, step)

            inter_next_q1_value = self.T1_Critic(inter_next_state, inter_next_action)
            inter_next_q2_valie = self.T2_Critic(inter_next_state, inter_next_action)

            inter_i_target = soft_i_return + self.gamma ** (step) * \
                            (torch.minimum(inter_next_q1_value, inter_next_q2_valie)
                             - self.alpha_explore * inter_next_logprob)
            intermediate_target[i, valid] = inter_i_target


        soft_return = n_step_return.clone()
        target = n_step_return.clone()
        for i in range(self.n_step_return):
            step = i + 1
            valid = gamma_pow == step

            if not valid.any():
                continue

            soft_i_return = n_step_return[valid].clone()
            for j in range(i):
                state = intermediate_state[valid, j, :]
                _, log_prob = self.Q_Actor.sample_repeat(state, step)
                soft_i_return -= (self.gamma) ** (j + 1) * self.alpha_explore * log_prob
            soft_return[valid] = soft_i_return
            target[valid] = soft_i_return
            
            valid_and_terminated = valid & ~terminated
            if not valid_and_terminated.any():
                continue
                
            next_i_state = next_state[valid_and_terminated]
            # normalize_next_state = self.preprocess(next_state)
            normalize_next_i_state = next_i_state
                
            next_i_action, next_i_log_prob = self.Q_Actor.sample_repeat(normalize_next_i_state, step)
            next_i_q1_value = self.T1_Critic(normalize_next_i_state, next_i_action)
            next_i_q2_value = self.T2_Critic(normalize_next_i_state, next_i_action)

            target[valid_and_terminated] = soft_return[valid_and_terminated] + (self.gamma ** step) * \
                             (torch.minimum(next_i_q1_value, next_i_q2_value) - self.alpha_explore * next_i_log_prob)
        
        return intermediate_target, target, intermediate_valid
        
    def update_q_critic(self, indx, state, action, reward, next_state, terminated, gamma_pow, 
                        intermediate_state, intermediate_action, intermediate_behavior_log_prob, 
                        intermediate_z, intermediate_n_return, probailities, ):
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        terminated = terminated.to(self.device)
        gamma_pow = gamma_pow.to(self.device)
        intermediate_state = intermediate_state.to(self.device)
        intermediate_action = intermediate_action.to(self.device)
        intermediate_behavior_log_prob = intermediate_behavior_log_prob.to(self.device)
        intermediate_z = intermediate_z.to(self.device)
        intermediate_n_return = intermediate_n_return.to(self.device)
        
        # normalize_state = self.preprocess(state)
        normalize_state = state
        q1_value = self.Q1_Critic(normalize_state, action)
        q2_value = self.Q2_Critic(normalize_state, action)
            
        intermediate_target, target, intermediate_valid = self.compute_target(next_state, reward, terminated, gamma_pow, 
                                                                              intermediate_state, intermediate_n_return)
        with torch.no_grad():
            batch_size, num_intermediate = intermediate_behavior_log_prob.shape[0], intermediate_behavior_log_prob.shape[1]
            steps = torch.arange(1, num_intermediate + 1, device=self.device)
            # нужна маска для правильно обработки концов траекторий, где количество шагов уже меньше заданного
            valid = steps.unsqueeze(0) < gamma_pow.unsqueeze(1)
            log_ratios = torch.zeros_like(intermediate_behavior_log_prob)
            
            if valid.any():
                current_log_prob = self.Q_Actor.get_gaussian_log_prob(intermediate_state[valid], intermediate_z[valid])
                log_ratios[valid] = current_log_prob - intermediate_behavior_log_prob[valid]
    
            prefix_log_rho = torch.cat([log_ratios.new_zeros((batch_size, 1)), log_ratios.cumsum(dim=1)], dim=1)
            # cumsum переносит log_rho дальше даже в случае окончания тректории, поэтому нужно снова применить маску
            prefix_valid = torch.cat([valid.new_ones((batch_size, 1)), valid], dim=1)
            masked_prefix_log_rho = prefix_log_rho.masked_fill(~prefix_valid, -torch.inf)
            
            valid_log_rho = prefix_log_rho[prefix_valid]
            sort_log_rho = valid_log_rho.sort().values
            quantile = 0.75
            position = quantile * (sort_log_rho.numel() - 1)
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                log_cap = sort_log_rho[lower]
            else:
                fraction = position - lower
                log_cap = torch.logaddexp(sort_log_rho[lower] + math.log(1 - fraction), 
                                          sort_log_rho[upper] + math.log(fraction))
                
            truncated_log_rho = torch.minimum(masked_prefix_log_rho, log_cap)
            max_truncated_log_rho = truncated_log_rho.max(dim=0, keepdim=True).values
            max_truncated_log_rho = torch.where(prefix_valid.any(dim=0, keepdim=True), 
                                                max_truncated_log_rho, torch.zeros_like(max_truncated_log_rho))
            
            normalized_truncated_log_rho = truncated_log_rho - max_truncated_log_rho        
            rho = normalized_truncated_log_rho.exp()

        all_targets = torch.cat([intermediate_target.T, target.unsqueeze(1)], dim=1)
        rows = torch.arange(batch_size, device=self.device)
        cols = gamma_pow - 1
        all_targets[rows, cols] = target

        num_valid = prefix_valid.sum(dim=1)
        delta_1 = q1_value.unsqueeze(1) - all_targets
        delta_2 = q2_value.unsqueeze(1) - all_targets
        masked_delta_1 = delta_1.masked_fill(~prefix_valid, 0.0)
        masked_delta_2 = delta_2.masked_fill(~prefix_valid, 0.0)
        weighted_delta_1 = rho * masked_delta_1
        weighted_delta_2 = rho * masked_delta_2
        effective_delta_1 = weighted_delta_1.sum(dim=1) / num_valid
        effective_delta_2 = weighted_delta_2.sum(dim=1) / num_valid
        buffer_prior = torch.maximum(torch.abs(effective_delta_1), 
                                     torch.abs(effective_delta_2)).detach().cpu().numpy()

        with torch.no_grad():
            td_abs = ((masked_delta_1.abs() + masked_delta_2.abs()).sum(dim=1) / (2 * num_valid)).mean()
            self.history['td_abs'].append(td_abs.item())

        critic_raw_loss = delta_1.square() + delta_2.square()
        
        weight = ((len(self.replay_buffer) * probailities) ** (-self.beta_per_weight)).to(self.device)
        weight = weight / weight.max()

        weighted_loss = weight.unsqueeze(1) * rho * critic_raw_loss
        masked_weighted_loss = weighted_loss.masked_fill(~prefix_valid, 0.0)
        critic_loss = (masked_weighted_loss.sum(dim=1) / prefix_valid.sum(dim=1)).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.replay_buffer.change_priority(indx, buffer_prior)

        return critic_loss.item()

    def update_q_actor(self, state):
        state = state.to(self.device)
        self.Q1_Critic.requires_grad_(False)
        self.Q2_Critic.requires_grad_(False)

        # normalize_state = self.preprocess(state)
        normalize_state = state
        _, policy_action, policy_log_prob = self.Q_Actor.sample(normalize_state)

        self.history['entropy'].append(-policy_log_prob.detach().mean().item())

        policy_q1_value = self.Q1_Critic(normalize_state, policy_action)
        policy_q2_value = self.Q2_Critic(normalize_state, policy_action)

        actor_loss = - (torch.minimum(policy_q1_value, policy_q2_value) - self.alpha_explore * policy_log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.Q1_Critic.requires_grad_(True)
        self.Q2_Critic.requires_grad_(True)
        return actor_loss.item()
        
    def fit(self):
        metric_time = 0
        log_every = 10000
        next_log_steps = (self.env_steps // log_every + 1) * log_every

        for episode in range(1, self.episodes+1):
            start_time = time.time()
            
            observation, info = self.env.reset()
            self.update_obs_metrics(observation)
            state = torch.tensor(observation, dtype=torch.float32)
            terminated = truncated = False  

            episod_return = 0
            episode_steps = 0
            while not (terminated or truncated):
                next_state, terminated, truncated, final_reward, total_reward = self.collect_data(state)
                state = next_state
                episod_return += total_reward
                episode_steps += 1
                self.env_steps += 1

                if len(self.replay_buffer) >= max(self.batch_size, self.buffer_size_for_start_train):
                    # batch_indx, batch_state, batch_action, batch_reward, batch_next_state, batch_terminated, \
                    #                          batch_gamma_pow, batch_intermediate_state, batch_probabilities, \
                    #                          batch_behavior_log_prob = self.replay_buffer.sample(self.batch_size)
                    batch = self.replay_buffer.sample(self.batch_size)
                    batch_state = batch[1]
                    critic_loss = self.update_q_critic(*batch)
                    self.history['critic_loss'].append(critic_loss)
                    self.update_steps += 1

                    if self.update_steps % self.policy_delay == 0:
                        actor_loss = self.update_q_actor(batch_state)
                        self.update_t_nets()
                        self.history['actor_loss'].append(actor_loss)

            self.history['return'].append(episod_return)

            max_steps = self.env.spec.max_episode_steps
            success = (not terminated and truncated and episode_steps >= max_steps)
            self.history['episode_success'].append(bool(success))
            self.history['episode_length'].append(episode_steps)

                
            end_time = time.time()
            metric_time += end_time - start_time
            
            if self.env_steps >= next_log_steps:
                while self.env_steps >= next_log_steps:
                    self.beta_per_weight = min(1, self.beta_per_weight * self.beta_per_weight_decay)
                    next_log_steps += log_every

                self.eval_policy()
                self.metrics(episode, metric_time)

    @torch.no_grad()
    def evaluate_for_visualization(self, seed=42):  

        eval_env = gym.make(self.env.spec.id, 
                            self.env.spec.max_episode_steps)

        was_training = self.Q_Actor.training
        self.Q_Actor.eval()

        try:
            observation, _ = eval_env.reset(seed=seed)
            truncated = terminated = False

            key = len(self.history['return'])
            record = {'actions': [], 'seed': seed, 'return': 0, 'length': 0}
        
            record['actions'] = []
            record['seed'] = seed
            episode_reward = episode_length = 0
        
            while not (truncated or terminated):
                state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
                # normalize_state = self.preprocess(state)
                
                action, _, _ = self.select_action(state, deterministic=True)
                observation, reward, terminated, truncated, _ = eval_env.step(action)
                            
                episode_reward += reward
                episode_length += 1
                
                record['actions'].append(action)
        finally:
            self.Q_Actor.train(was_training)
            eval_env.close()
            
        record['return'] = episode_reward 
        record['length'] = episode_length
        self.history['eval_visual'][key] = record

    @torch.no_grad()
    def evaluate_policy(self, seeds=(42, 43, 44, 45, 46)):
        eval_env = gym.make(
            self.env.spec.id,
            max_episode_steps=self.env.spec.max_episode_steps,
        )
        returns = []
        was_training = self.Q_Actor.training
        self.Q_Actor.eval()
        try:
            for seed in seeds:
                observation, _ = eval_env.reset(seed=seed)
                terminated = truncated = False
                episode_return = 0.0
                while not (terminated or truncated):
                    state = torch.as_tensor(observation, dtype=torch.float32)
                    action, _, _ = self.select_action(state, deterministic=True)
                    observation, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_return += reward
                returns.append(episode_return)
        finally:
            self.Q_Actor.train(was_training)
            eval_env.close()

        record = {
            'env_steps': self.env_steps,
            'seeds': list(seeds),
            'returns': returns,
            'return_mean': float(np.mean(returns)),
            'return_std': float(np.std(returns)),
        }
        self.history['eval_summary'].append(record)
        return record

    def metrics(self, episode, metric_time):
        recent_success = self.history['episode_success'][-100:]
        success_rate = np.mean(recent_success)
        self.history['success_rate'].append(success_rate)
        recent_length = self.history['episode_length'][-100:]
        recent_return = self.history['return'][-100:]
        print(f'episod: {episode}, env steps: {self.env_steps}') 
        print(f'success rate (last 100 episodes): {success_rate:.1%}')
        print(f'mean_episode_length (last 100 episodes): {np.mean(recent_length)}')
        print(f'mean return (last 100 episodes): {np.mean(recent_return)}')
        if self.history['td_abs']:
            print(f"td_abs (last 1000 critic updates): {np.mean(self.history['td_abs'][-1000:]):.4f}")
        if self.history['entropy']:
            print(f"entropy (last 1000 actor updates): {np.mean(self.history['entropy'][-1000:]):.4f}")
        if self.history["eval_summary"]:
            result = self.history["eval_summary"][-1]
            print(
                f"eval return: {result['return_mean']:.1f} "
                f"+/- {result['return_std']:.1f}"
            )
        print(f'time: {metric_time // 60} min')
        print('-'*50)