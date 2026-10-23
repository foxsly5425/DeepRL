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
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return  3 * F.tanh(self.net(x))

class critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
            
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return  self.net(x).squeeze(-1)

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, alpha, epsilon, max_priority=1.0):
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

        self.max_priority = max_priority
        self.alpha = alpha
        self.epsilon = epsilon
        
    def append(self, data):
        state, action, reward, next_state, terminated, gamma_pow = data

        self.states[self.last_pos] = np.asarray(state, dtype=np.float32)
        self.actions[self.last_pos] = action
        self.rewards[self.last_pos] = reward
        self.next_states[self.last_pos] = next_state
        self.terminated[self.last_pos] = terminated
        self.priorities[self.last_pos] = self.max_priority
        self.gamma_pow[self.last_pos] = gamma_pow

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
            torch.as_tensor(probabilities[indx], dtype=torch.float32),
        )

    def __len__(self):
        return self.size
        
class DDPG:
    def __init__(self, env, episodes=10000, batch_size=64, device='cpu',
                gamma=0.99, lr=1e-3, buffer_size_for_start_train=2000,
                max_size_buffer=40000, alpha_buffer=0.5, epsilon_buffer=1e-4, beta_per_weight=0.5, beta_per_weight_decay=1.05,
                n_step_return=1, tau_update_t_nets=0.005):

        self.n_step_return = n_step_return 
        self.n_step_buffer = deque()

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        self.device = device
        
        self.Q_Actor = actor(self.state_dim, self.action_dim).to(self.device)
        self.Q_Critic = critic(self.state_dim, self.action_dim).to(self.device)
        
        self.T_Actor = actor(self.state_dim, self.action_dim).to(self.device)
        self.T_Critic = critic(self.state_dim, self.action_dim).to(self.device)

        self.T_Actor.load_state_dict(self.Q_Actor.state_dict())
        self.T_Critic.load_state_dict(self.Q_Critic.state_dict())

        self.T_Actor.requires_grad_(False)
        self.T_Critic.requires_grad_(False)
        self.T_Actor.eval()
        self.T_Critic.eval()

        self.tau_update_t_nets = tau_update_t_nets
        
        self.gamma = gamma
        self.loss_fn = nn.MSELoss(reduction='none')
        self.actor_optimizer = torch.optim.Adam(self.Q_Actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.Q_Critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(max_size=max_size_buffer, state_dim=self.state_dim, action_dim=self.action_dim,
                                          alpha=alpha_buffer, epsilon=epsilon_buffer)
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
                        'episode_success': [], 'episode_length': []}

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
    def select_action(self, state, noise_std=0.1):
        state = state.to(self.device)
        action = self.Q_Actor(state).cpu().numpy()
        noise = np.random.normal(loc=0.0, scale=noise_std, size=action.shape)
        explor_action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)
        return explor_action.astype(np.float32)

    def n_step_sliding_window(self):
        n_return = gamma_pow = 0
        first_state, first_action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        for i in range(self.n_step_return):
            _, _, reward, next_state, terminated, truncated = self.n_step_buffer[i]
            n_return += (self.gamma ** gamma_pow) * reward
            gamma_pow += 1

            if truncated or terminated:
                break

        self.replay_buffer.append([first_state, first_action, n_return, next_state, terminated, gamma_pow])
        self.n_step_buffer.popleft()

    def collect_data(self, state):        
        action = self.select_action(state)

        final_reward = -100
        total_reward = 0

        next_observation, reward, terminated, truncated, _ = self.env.step(action)
        self.update_obs_metrics(next_observation)

        next_state = torch.tensor(next_observation, dtype=torch.float32)

        self.n_step_buffer.append([state, action, reward, next_state, terminated, truncated])

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
        for online_net, target_net in ((self.Q_Actor, self.T_Actor), (self.Q_Critic, self.T_Critic)):
            for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
                target_param.mul_(1 - self.tau_update_t_nets)
                target_param.add_(online_param, alpha=self.tau_update_t_nets)

    @torch.no_grad()
    def compute_target(self, next_state, n_step_return, terminated, gamma_pow):
        normalize_next_state = self.preprocess(next_state)
        next_action = self.T_Actor(normalize_next_state)
        next_q_value = self.T_Critic(normalize_next_state, next_action)
        target = torch.where(terminated, n_step_return, n_step_return + (self.gamma ** gamma_pow) * next_q_value)
        return target

    def update_q_critic(self, indx, state, action, reward, next_state, terminated, gamma_pow, probailities):
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        terminated = terminated.to(self.device)
        gamma_pow = gamma_pow.to(self.device)

        normalize_state = self.preprocess(state)
        q_value = self.Q_Critic(normalize_state, action)
        target = self.compute_target(next_state, reward, terminated, gamma_pow)
        critic_raw_loss = self.loss_fn(q_value, target)
        
        buffer_prior = torch.abs(target - q_value).detach().cpu().numpy()
        weight = ((len(self.replay_buffer) * probailities) ** (-self.beta_per_weight)).to(self.device)
        weight = weight / weight.max()

        critic_loss = (weight * critic_raw_loss).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.replay_buffer.change_priority(indx, buffer_prior)

        return critic_loss.item()

    def update_q_actor(self, state):
        state = state.to(self.device)
        self.Q_Critic.requires_grad_(False)

        normalize_state = self.preprocess(state)
        policy_action = self.Q_Actor(normalize_state)
        policy_q_value = self.Q_Critic(state, policy_action)
        actor_loss = -policy_q_value.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.Q_Critic.requires_grad_(True)
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
                    batch_indx, batch_state, batch_action, batch_reward, batch_next_state, batch_terminated, batch_gamma_pow, batch_probabilities = self.replay_buffer.sample(self.batch_size)
                    critic_loss = self.update_q_critic(batch_indx, batch_state, batch_action, batch_reward, batch_next_state, batch_terminated, batch_gamma_pow, batch_probabilities)
                    actor_loss = self.update_q_actor(batch_state)
                    self.update_t_nets()
                    
                    self.history['critic_loss'].append(critic_loss)
                    self.history['actor_loss'].append(actor_loss)
                    
                    self.update_steps += 1

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

    def metrics(self, episode, metric_time):
        recent_success = self.history['episode_success'][-100:]
        success_rate = 100 * np.mean(recent_success)
        self.history['success_rate'].append(success_rate)
        recent_length = self.history['episode_length'][-100:]
        recent_return = self.history['return'][-100:]
        print(f'episod: {episode}, success rate: {success_rate}%')
        print(f'mean_episode_length: {np.mean(recent_length)}')
        print(f'mean return: {np.mean(recent_return) / 100}')
        print(f'time: {metric_time // 60} min')
        print('-'*50)