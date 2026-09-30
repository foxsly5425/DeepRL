import sys, os

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import random
import time

from collections import deque


import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="rgb_array")
env.reset()

plt.imshow(env.render())
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

if torch.xpu.is_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

print("Device:", device)

class Q_net(nn.Module):
    def __init__(self, state_dim, action_dim, dueling_dqn_flag):
        super().__init__()
        self.dueling_dqn_flag = dueling_dqn_flag

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        if self.dueling_dqn_flag:
            self.v_head = nn.Linear(64, 1)
            self.adv_head = nn.Linear(64, action_dim)

            nn.init.normal_(self.v_head.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.v_head.bias, -1.0)
            nn.init.normal_(self.adv_head.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.adv_head.bias, -1.0)
        else:
            self.head = nn.Linear(64, action_dim)
            
            nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.head.bias, -1.0)

    def forward(self, x):
        x = self.net(x)
        if self.dueling_dqn_flag:
            v = self.v_head(x)
            adv = self.adv_head(x)
            mean_adv = adv.mean(dim=-1, keepdim=True)
            outp = v + adv - mean_adv
        else:
            outp = self.head(x)
        return outp

class ReplayBuffer:
    def __init__(self, max_size, state_dim, alpha, epsilon, max_priority=1.0):
        self.max_size = max_size
        self.last_pos = 0
        self.size = 0

        self.states = np.empty((self.max_size, state_dim), dtype=np.float32)
        self.actions = np.empty(self.max_size, dtype=np.int64)
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
        
class DQN:
    def __init__(self, env, episodes=10000, batch_size=64, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.95, device='cpu', 
                exploration_repeat=20, gamma=0.99, lr=1e-3, target_update_interval=1000, buffer_size_for_start_train=2000, 
                max_size_buffer=40000, alpha_buffer=0.5, epsilon_buffer=1e-4, beta_per_weight=0.5, beta_per_weight_decay=1.05,
                double_dqn_flag=False, dueling_dqn_flag=False, n_step_return=1):
        
        self.double_dqn_flag = double_dqn_flag
        self.dueling_dqn_flag = dueling_dqn_flag

        self.n_step_return = n_step_return 
        self.n_step_buffer = deque()

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        self.device = device
        
        self.q_net = Q_net(self.state_dim, self.action_dim, self.dueling_dqn_flag).to(self.device)
        
        self.t_net = Q_net(self.state_dim, self.action_dim, self.dueling_dqn_flag).to(self.device)
        self.t_net.load_state_dict(self.q_net.state_dict())
        self.t_net.requires_grad_(False)
        self.t_net.eval()
        
        self.gamma = gamma
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(max_size=max_size_buffer, state_dim=self.state_dim, alpha=alpha_buffer, epsilon=epsilon_buffer)
        self.beta_per_weight = beta_per_weight
        self.beta_per_weight_decay = beta_per_weight_decay
        self.exploration_repeat = exploration_repeat
        self.buffer_size_for_start_train = buffer_size_for_start_train
        
        self.episodes = episodes
        self.batch_size = batch_size
        self.update_steps = 0
        self.target_update_interval = target_update_interval
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.obs_low  = torch.tensor(self.env.observation_space.low, dtype=torch.float32, device=self.device)
        self.obs_high = torch.tensor(self.env.observation_space.high, dtype=torch.float32, device=self.device)

        self.history = {'return': [], 'loss': [], 'success_rate': [], 'epsilon': []}

    def preprocess(self, obs):
        norm_obs = 2 * (obs - self.obs_low) / (self.obs_high - self.obs_low) - 1
        return norm_obs

    def select_action(self, state):
        if torch.rand(1) < self.epsilon:
            return self.env.action_space.sample(), True

        with torch.no_grad():
            state = state.to(self.device)
            action = self.q_net(self.preprocess(state)).argmax().item()

        return action, False

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
        action, random_action_flag = self.select_action(state)

        final_reward = -100
        total_reward = 0

        next_observation, reward, terminated, truncated, _ = self.env.step(action)
        next_state = torch.tensor(next_observation)

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

        return next_state, terminated, truncated, final_reward, total_reward, random_action_flag


    def update_t_net(self):
        self.t_net.load_state_dict(self.q_net.state_dict())

    @torch.no_grad()
    def compute_target(self, next_state, n_step_return, terminated, gamma_pow):
        if not self.double_dqn_flag:
            next_q_vector = self.t_net(self.preprocess(next_state))
            next_q_value = next_q_vector.max(dim=-1).values
        else:
            next_q_vector = self.q_net(self.preprocess(next_state))
            action = next_q_vector.argmax(dim=-1).unsqueeze(1)
            next_q_value = self.t_net(self.preprocess(next_state)).gather(dim=1, index=action).squeeze(1)
        target = torch.where(terminated, n_step_return, n_step_return + (self.gamma ** gamma_pow) * next_q_value)
                        
        return target

    def update_q_net(self, indx, state, action, reward, next_state, terminated, gamma_pow, probailities):
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        terminated = terminated.to(self.device)
        gamma_pow = gamma_pow.to(self.device)

        q_value = self.q_net(self.preprocess(state))
        q_value = q_value.gather(dim=1, index=action.unsqueeze(1)).squeeze(1)
        target = self.compute_target(next_state, reward, terminated, gamma_pow)
        raw_loss = self.loss_fn(q_value, target)
        
        buffer_prior = torch.abs(target - q_value).detach().cpu().numpy()
        weight = ((len(self.replay_buffer) * probailities) ** (-self.beta_per_weight)).to(self.device)
        weight = weight / weight.max()

        loss = (weight * raw_loss).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.replay_buffer.change_priority(indx, buffer_prior)

        return loss.item()

    def fit(self):
        max_position_lst = []
        min_position_lst = []
        metric_time = 0
        success_rate = 0
        
        for episode in range(1, self.episodes+1):
            start_time = time.time()
            
            observation, info = self.env.reset()
            state = torch.tensor(observation)
            terminated = truncated = False  
            max_position = min_position = observation[0]

            episod_return = 0
            while not (terminated or truncated):
                next_state, terminated, truncated, final_reward, total_reward, random_action_flag = self.collect_data(state)
                state = next_state
                episod_return += total_reward

        
                if len(self.replay_buffer) >= max(self.batch_size, self.buffer_size_for_start_train):
                    batch_indx, batch_state, batch_action, batch_reward, batch_next_state, batch_terminated, batch_gamma_pow, batch_probabilities = self.replay_buffer.sample(self.batch_size)
                    loss = self.update_q_net(batch_indx, batch_state, batch_action, batch_reward, batch_next_state, batch_terminated, batch_gamma_pow, batch_probabilities)
                    
                    self.history['loss'].append(loss)
                    
                    self.update_steps += 1

                    if self.update_steps % self.target_update_interval == 0:
                        self.update_t_net()
 
            self.history['return'].append(episod_return)

            if terminated and final_reward == 100:
                    success_rate += 1                

            if episode % 100 == 0 and success_rate > 10:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % 100 == 0:
                self.beta_per_weight = min(1, self.beta_per_weight * self.beta_per_weight_decay)

                
            end_time = time.time()
            metric_time += end_time - start_time
            
            if episode % 100  == 0:
                self.metrics(episode, max_position_lst, min_position_lst, metric_time, success_rate)
                self.history['success_rate'].append(success_rate)
                self.history['epsilon'].append(self.epsilon)
                success_rate = 0

    def metrics(self, episode, max_position_lst, min_position_lst, metric_time, success_rate):
        print(f'episod: {episode}, success rate: {success_rate}%')
        print(f'mean return: {sum(self.history['return'][-100:]) / 100}')
        print(f'epsilon: {self.epsilon}, time: {metric_time // 60} min')
        print('-'*50)