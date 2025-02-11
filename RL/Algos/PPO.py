import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import random
import pickle
import copy
import datetime
import logging
import threading
import sys
import os
import csv
import warnings
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

tf.config.optimizer.set_jit(True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', message='Skipping variable loading for optimizer')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')

tfd = tfp.distributions
tfk = tfp.math.psd_kernels

class PPO():
    def __init__(self, env, state_size, action_size, max_steps=1000000, rollout_steps=5000, score_steps=5000, num_workers=5, num_players=5, maxlen=10000, batch_size=1024, gamma=0.99, env_name='InvertedPendulum'):
        self.name = 'PPO'
        self.env_name= env_name
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.envs = [copy.deepcopy(env) for _ in range(num_workers)]
        self.num_workers = num_workers
        self.num_players = num_players
        self.buffer = []
        self.maxlen = maxlen
        self.max_steps = max_steps
        self.ep = 0
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.score_steps = score_steps
        self.gamma = gamma
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        self.epochs_per_episode = 10
        self.actor_losses = []
        self.critic_losses = []
        self.episode_rewards = []
        self.reward_each_episode = []
        self.cum_avg_rewards = []
        self.score_each_episode = []
        self.cum_avg_score = []
        self.steps = []
        self.total_steps = 0
        
        self.save_folder = self.env_name + '-' + self.name + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.create_save_file()
        
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = Adam(1e-4, clipnorm=0.5)
        self.critic_optimizer = Adam(1e-4, clipnorm=0.5)
        self.actor_file = self.save_folder + '/actor.keras'
        self.critic_file = self.save_folder + '/critic.keras'
        
        # plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        
        self.log_file = self.save_folder + '/logs.txt'
        self.setup_logging()
        
    def setup_logging(self):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        log_filename = self.log_file
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(message)
        
    def create_save_file(self):
        save_header = ['Timesteps','Avg Training Steps','Avg Training Reward','Actor Loss','Critic Loss','Avg Score Steps','Avg Score Reward']
        
        with open(self.save_folder + '/info.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(save_header)
        
    def save_info(self, timesteps, avg_train_steps, avg_train_rewards, actor_loss, critic_loss, avg_score_steps, avg_score_reward):
        new_info = [timesteps, avg_train_steps, avg_train_rewards, actor_loss, critic_loss, avg_score_steps, avg_score_reward]
        
        with open(self.save_folder + '/info.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_info)
        
    def save_info_async(self, timesteps, avg_train_steps, avg_train_rewards, actor_loss, critic_loss, avg_score_steps, avg_score_reward):
        thread = threading.Thread(target=self.save_info, args=(timesteps, avg_train_steps, avg_train_rewards, actor_loss, critic_loss, avg_score_steps, avg_score_reward))
        thread.start()
    
    def build_actor(self):
        state_input = Input(shape=(self.state_size,), dtype=tf.float32)
        x = Dense(64, activation='relu')(state_input)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        mean_output = Dense(self.action_size, activation='tanh')(x)
        log_std_output = Dense(self.action_size, activation='linear')(x) + 1e-8
        
        model = Model(inputs=state_input, outputs=[mean_output,log_std_output])
        return model
    
    def build_critic(self):
        state_input = Input(shape=(self.state_size,), dtype=tf.float32)
        x = Dense(64, activation='relu')(state_input)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='linear')(x)
        
        model = Model(inputs=state_input, outputs=output)
        return model
    
    def save_models(self):
        self.actor.save(self.actor_file, include_optimizer=False)
        self.critic.save(self.critic_file, include_optimizer=False)
        
    def save_models_async(self):
        thread = threading.Thread(target=self.save_models, args=())
        thread.start()
    
    def load_models(self):
        self.actor = keras.models.load_model(self.actor_file)
        self.critic = keras.models.load_model(self.critic_file)
    
    def compute_gae(self, rewards, dones, values):
        advantages = np.zeros_like(rewards)
        targets = np.zeros_like(rewards)
        next_adv = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + self.gamma * values[-1] * (1 - dones[t]) - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * 0.95 * next_adv * (1 - dones[t])
            next_adv = advantages[t]

        targets = advantages + values
        return advantages, targets
    
    def collect_rollouts(self):
        rollouts = []
        ep_rewards = np.zeros(self.num_workers)
        total_ep_rewards = []
        steps = np.zeros(self.num_workers)
        ep_steps = np.zeros(self.num_workers)
        total_ep_steps = []
        dones = np.zeros(self.num_workers, dtype=bool)
        states = np.array([e.reset()[0] for e in self.envs])
        eps = 0
        
        while np.sum(steps) <= self.rollout_steps:
            ep_steps += 1
            steps += 1
            mean, log_std = self.actor(np.atleast_2d(states), training=False)
            std = tf.exp(log_std)
            dist = tfp.distributions.Normal(mean, std)
            actions = dist.sample()
            actions = tf.clip_by_value(actions, self.env.action_min, self.env.action_max)
            log_probs = tf.reduce_sum(dist.log_prob(actions), axis=-1).numpy()
            actions = actions.numpy()
            print(actions)
            nextS, rewards, dones, _, _ = zip(*[e.step(action) for e, action in zip(self.envs, actions)])
            nextS = np.array(nextS)
            rewards = np.array(rewards)
            dones = np.array(dones)
            ep_rewards += rewards
            for i in range(self.num_workers):
                rollouts.append((states[i], actions[i], rewards[i], nextS[i], dones[i], log_probs[i]))
                if dones[i]:
                    states[i] = self.envs[i].reset()[0]
                    total_ep_rewards.append(ep_rewards[i])
                    total_ep_steps.append(ep_steps[i])
                    ep_rewards[i] = 0
                    ep_steps[i]= 0
            states = nextS
        
        self.total_steps += np.sum(steps)
            
        self.buffer.extend(rollouts)
        if len(self.buffer) > self.maxlen:
            self.buffer = self.buffer[-self.maxlen:]
            
        return np.mean(ep_steps), np.mean(total_ep_rewards)
    
    def train_models(self):
        for _ in range(self.epochs_per_episode):
            batch = np.array(self.buffer, dtype=object)[np.random.choice(len(self.buffer), min(len(self.buffer),self.batch_size), replace=False)]
            batch_states = np.stack(batch[:, 0]).astype(np.float32)
            batch_actions = np.stack(batch[:, 1]).astype(np.float32)
            batch_rewards = np.stack(batch[:, 2]).astype(np.float32)
            batch_nextS = np.stack(batch[:, 3]).astype(np.float32)
            batch_dones = np.stack(batch[:, 4]).astype(np.float32)
            batch_log_probs = np.stack(batch[:, 5]).astype(np.float32)
            
            values = self.critic(batch_states, training=False)
            batch_advantages, batch_targets = self.compute_gae(batch_rewards, batch_dones, values)
            
            batch_advantages = (batch_advantages - np.mean(batch_advantages)) / (np.std(batch_advantages) + 1e-8)
                
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                mean, log_std = self.actor(batch_states, training=True)
                std = tf.exp(log_std)
                dist = tfp.distributions.Normal(mean, std)
                log_probs = tf.reduce_sum(dist.log_prob(batch_actions), axis=-1, keepdims=True)

                log_ratio = log_probs - batch_log_probs
                ratio = tf.exp(log_ratio)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -tf.reduce_mean(tf.minimum(ratio * batch_advantages, clipped_ratio * batch_advantages))
                
                values = self.critic(batch_states, training=True)
                value_loss = tf.reduce_mean(tf.square(batch_targets - values))
                
                entropy_loss = -tf.reduce_mean(dist.entropy())
                actor_loss = policy_loss - self.entropy_coef * entropy_loss
                critic_loss = value_loss
                total_loss = actor_loss + critic_loss
                
                approx_kl = tf.reduce_mean((tf.exp(log_ratio) - 1) - log_ratio)
                
                
            actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            
        
        self.save_models_async()
        
        return actor_loss.numpy(), critic_loss.numpy()
        
    
    def learn(self):
        while self.total_steps <= self.max_steps:
            t_steps, t_rewards = self.collect_rollouts()
            a_loss, c_loss = self.train_models()
            s_steps, s_rewards = self.score()
            
            print(f'\nTimesteps: {int(self.total_steps)} | Reward: {s_rewards:.2f} | Steps: {int(s_steps)}')
            self.log(f'\nTimesteps: {int(self.total_steps)} | Reward: {s_rewards:.2f} | Steps: {int(s_steps)}')
            
            self.save_info_async(int(self.total_steps), t_steps, t_rewards, a_loss, c_loss, s_steps, s_rewards)
        
    def plot_rewards(self):
        self.ax.clear()
        self.ax.plot(self.cum_avg_rewards)
        self.ax.grid(False)
        self.ax.set_xlabel('Episodes')
        self.ax.set_ylabel('Reward')
        self.ax.set_title(self.env_name + '-' + self.name)
        plt.savefig(self.save_folder + '/rewards.jpg')
        
    def plot_async(self):
        thread = threading.Thread(target=self.plot_rewards)
        thread.start()
    
    def score(self):
        rollouts = []
        r = 0
        steps = np.zeros(self.num_players)
        ep_steps = np.zeros(self.num_players)
        total_ep_steps= []
        dones = np.zeros(self.num_players, dtype=bool)
        states = np.array([e.reset()[0] for e in self.envs])
        ep_rewards = np.zeros(self.num_players)
        total_ep_rewards = []
        
        while np.sum(steps) <= self.score_steps:
            ep_steps += 1
            steps += 1
            mean, log_std = self.actor(np.atleast_2d(states), training=False)
            std = tf.exp(log_std)
            dist = tfp.distributions.Normal(mean, std)
            actions = dist.sample()
            actions = tf.clip_by_value(actions, self.env.action_min, self.env.action_max)
            actions = actions.numpy()
            nextS, rewards, dones, _, _ = zip(*[e.step(action) for e, action in zip(self.envs, actions)])
            nextS = np.array(nextS)
            rewards = np.array(rewards)
            dones = np.array(dones)
            ep_rewards += rewards
            for i in range(self.num_players):
                if dones[i]:
                    states[i] = self.envs[i].reset()[0]
                    total_ep_rewards.append(ep_rewards[i])
                    total_ep_steps.append(ep_steps[i])
                    ep_rewards[i] = 0
                    ep_steps[i] = 0
            states = nextS
            
        self.reward_each_episode.extend(total_ep_rewards)
        self.steps.extend(total_ep_steps)
        self.cum_avg_rewards.append(np.mean(self.reward_each_episode))
        self.plot_async()
        
        return np.mean(total_ep_steps), np.mean(total_ep_rewards)
    
        