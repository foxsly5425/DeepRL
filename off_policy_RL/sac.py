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
        return action, log_prob

    def get_log_prob(self, state, action, eps=1e-6):
        distribution = self.compute_distribution(state)
        normalized_action = (action - self.action_bias) / self.action_scale 
        normalized_action = normalized_action.clamp(-1+eps, 1-eps)
        z = torch.atanh(normalized_action)
        log_prob = self.compute_log_prob(distribution, z)
        return log_prob
        

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
        self.max_priority = max_priority
        self.alpha = alpha
        self.epsilon = epsilon
        
    def append(self, data):
        state, action, reward, next_state, terminated, gamma_pow, \
        intermediate_state, intermediate_action, intermediate_behavior_log_prob = data

        self.states[self.last_pos] = np.asarray(state, dtype=np.float32)
        self.actions[self.last_pos] = action
        self.rewards[self.last_pos] = reward
        self.next_states[self.last_pos] = next_state
        self.terminated[self.last_pos] = terminated
        self.priorities[self.last_pos] = self.max_priority
        self.gamma_pow[self.last_pos] = gamma_pow
        self.intermediate_state[self.last_pos] = intermediate_state
        self.intermediate_action[self.last_pos] = intermediate_action
        self.intermediate_behavior_log_prob[self.last_pos] = intermediate_behavior_log_prob
        
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
            torch.as_tensor(probabilities[indx], dtype=torch.float32),
        )

    def __len__(self):
        return self.size
        
class SAC:
    def __init__(self, env, episodes=10000, batch_size=64, device='cpu', 
                gamma=0.99, lr=1e-3, buffer_size_for_start_train=2000, 
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

        self.obs_count = 0
        self.obs_mean = np.zeros(self.state_dim, dtype=np.float64)
        self.obs_m2 = np.zeros(self.state_dim, dtype=np.float64)

        self.history = {'return': [], 'critic_loss': [], 'actor_loss': [], 'success_rate': [],
                        'episode_success': [], 'episode_length': [], 'eval': dict()}

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
            behavior_log_prob = None
        else:
            action, behavior_log_prob = self.Q_Actor.sample(normalize_state)
            behavior_log_prob.cpu().numpy().astype(np.float32)
        return action.cpu().numpy().astype(np.float32), behavior_log_prob

    @torch.no_grad()
    def n_step_sliding_window(self):
        n_return = gamma_pow = 0
        first_state, first_action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        intermediate_action = np.zeros((self.n_step_return - 1, self.action_dim), dtype=np.float32)
        intermediate_behavior_log_prob = np.zeros(self.n_step_return - 1, dtype=np.float32)
        intermediate_state = np.zeros((self.n_step_return - 1, self.state_dim), dtype=np.float32)
                                      
        for i in range(self.n_step_return):
            state, action, reward, next_state, terminated, truncated, behavior_log_prob = self.n_step_buffer[i]
            n_return += (self.gamma ** gamma_pow) * reward
            gamma_pow += 1

            if i > 0:
                intermediate_action[i-1] = np.asarray(action, dtype=np.float32)
                intermediate_behavior_log_prob[i-1] = np.asarray(behavior_log_prob, dtype=np.float32)
                intermediate_state[i-1] = np.asarray(state, dtype=np.float32)

            if truncated or terminated:
                break

        self.replay_buffer.append([first_state, first_action, n_return, next_state, terminated, gamma_pow, 
                                   intermediate_state, intermediate_action, intermediate_behavior_log_prob])
        self.n_step_buffer.popleft()

    def collect_data(self, state):        
        action, behavior_log_prob = self.select_action(state)
            
        final_reward = -100
        total_reward = 0

        next_observation, reward, terminated, truncated, _ = self.env.step(action)
        self.update_obs_metrics(next_observation)
        
        next_state = torch.tensor(next_observation, dtype=torch.float32)

        self.n_step_buffer.append([state, action, reward, next_state, terminated, truncated, behavior_log_prob])

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
    def compute_target(self, next_state, n_step_return, terminated, gamma_pow, intermediate_state):
        # normalize_next_state = self.preprocess(next_state)
        normalize_next_state = next_state
        soft_return = n_step_return.clone()
        
        for j in range(intermediate_state.shape[1]):
            step = j + 1
            valid = gamma_pow > step

            if not valid.any():
                continue
            
            state = intermediate_state[valid, j, :]
            action, log_prob = self.Q_Actor.sample(state)
            
            soft_return[valid] -= self.gamma ** (step) * self.alpha_explore * log_prob
            
        next_action, next_log_prob = self.Q_Actor.sample(normalize_next_state)
        next_q1_value = self.T1_Critic(normalize_next_state, next_action)
        next_q2_value = self.T2_Critic(normalize_next_state, next_action)
        
        target = torch.where(terminated, soft_return, soft_return + (self.gamma ** gamma_pow) * 
                             (torch.minimum(next_q1_value, next_q2_value) - self.alpha_explore * next_log_prob))
        return target

    def update_q_critic(self, indx, state, action, reward, next_state, terminated, 
                        gamma_pow, intermediate_state, intermediate_action, 
                        intermediate_behavior_log_prob, probailities, ):
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        terminated = terminated.to(self.device)
        gamma_pow = gamma_pow.to(self.device)
        intermediate_state = intermediate_state.to(self.device)
        intermediate_action = intermediate_action.to(self.device)
        intermediate_behavior_log_prob = intermediate_behavior_log_prob.to(self.device)

        with torch.no_grad():
            steps = torch.arange(1, intermediate_behavior_log_prob.shape[1] + 1, device=self.device)
            valid = steps.unsqueeze(0) < gamma_pow.unsqueeze(1)
            log_ratios = torch.zeros_like(intermediate_behavior_log_prob)
            
            if valid.any():
                current_log_prob = self.Q_Actor.get_log_prob(intermediate_state[valid], intermediate_action[valid])
                log_ratios[valid] = current_log_prob - intermediate_behavior_log_prob[valid]

            log_rho = log_ratios.sum(dim=1)
            q = 0.75

            sorted_logs = log_rho.sort().values
            position = q * (sorted_logs.numel() - 1)
            lower = math.floor(position)
            upper = math.ceil(position)

            if lower == upper:
                log_cap = sorted_logs[lower]
            else:
                fraction = position - lower
                log_cap = torch.logaddexp(sorted_logs[lower] + math.log(1 - fraction), 
                                          sorted_logs[upper] + math.log(fraction))
                
            clipped_log_rho = torch.minimum(log_rho, log_cap)
            normalized_clipped_log_rho = clipped_log_rho - clipped_log_rho.max()
            rho = normalized_clipped_log_rho.exp()
            
        intermediate_current_log_prob = self.Q_Actor.get_log_prob(intermediate_state, intermediate_action)
        intermediate_pho = torch.exp(intermediate_current_log_prob - intermediate_behavior_log_prob)
        
        # normalize_state = self.preprocess(state)
        normalize_state = state
        q1_value = self.Q1_Critic(normalize_state, action)
        q2_value = self.Q2_Critic(normalize_state, action)

        target = self.compute_target(next_state, reward, terminated, gamma_pow, intermediate_state)
        critic_raw_loss = self.loss_fn(q1_value, target) + self.loss_fn(q2_value, target)
        
        buffer_prior = torch.maximum(torch.abs(target - q1_value), 
                                     torch.abs(target - q2_value)).detach().cpu().numpy()
        weight = ((len(self.replay_buffer) * probailities) ** (-self.beta_per_weight)).to(self.device)
        weight = weight / weight.max()

        critic_loss = (weight * rho * critic_raw_loss).mean()
        
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
        policy_action, policy_log_prob = self.Q_Actor.sample(normalize_state)
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
        max_position_lst = []
        min_position_lst = []
        metric_time = 0
        
        for episode in range(1, self.episodes+1):
            start_time = time.time()
            
            observation, info = self.env.reset()
            self.update_obs_metrics(observation)
            state = torch.tensor(observation, dtype=torch.float32)
            terminated = truncated = False  
            max_position = min_position = observation[0]

            episod_return = 0
            episode_steps = 0
            while not (terminated or truncated):
                next_state, terminated, truncated, final_reward, total_reward = self.collect_data(state)
                state = next_state
                episod_return += total_reward
                episode_steps += 1
        
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

            if episode % 100 == 0:
                self.beta_per_weight = min(1, self.beta_per_weight * self.beta_per_weight_decay)

                
            end_time = time.time()
            metric_time += end_time - start_time
            
            if episode % 100  == 0:
                self.metrics(episode, metric_time)

    @torch.no_grad()
    def eval(self, seed=42):    
        observation, _ = self.env.reset(seed=seed)
        truncated = terminated = False

        key = len(self.history['return'])
        record = {'actions': [], 'seed': seed, 'return': 0, 'length': 0}
        
        record['actions'] = []
        record['seed'] = seed
        episode_reward = episode_length = 0
        
        while not (truncated or terminated):
            state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            normalize_state = self.preprocess(state)
            
            action = self.Q_Actor(normalize_state, deterministic=True).cpu().numpy()
            observation, reward, terminated, truncated, _ = self.env.step(action)
                        
            episode_reward += reward
            episode_length += 1
            
            record['actions'].append(action)
            
        record['return'] = episode_reward 
        record['length'] = episode_length
        self.history['eval'][key] = record

    def metrics(self, episode, metric_time):
        recent_success = self.history['episode_success'][-100:]
        success_rate = 100 * np.mean(recent_success)
        self.history['success_rate'].append(success_rate)
        recent_length = self.history['episode_length'][-100:]
        recent_return = self.history['return'][-100:]
        print(f'episod: {episode}, success rate: {success_rate}%')
        print(f'mean_episode_length: {np.mean(recent_length)}')
        print(f'mean return: {np.mean(recent_return)}')
        print(f'time: {metric_time // 60} min')
        print('-'*50)