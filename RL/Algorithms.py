import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import random
import time
import pickle
import copy
import datetime
import logging
import threading
import sys
from multiprocessing import Pool
import os
import warnings
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

tf.config.optimizer.set_jit(True)
tf.config.run_functions_eagerly(True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', message='Skipping variable loading for optimizer')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')

class Buffer():
    def __init__(self, maxlen, algo):
        self.algo = algo
        self.maxlen = maxlen
        self.len = 0
        self.states = np.array([])
        self.actions = np.array([])
        self.rewards = np.array([])
        self.nextS = np.array([])
        self.dones = np.array([])
        self.log_prob = np.array([])
    
    def append(self, transition, algo):
        self.len += 1
        self.states = np.append(self.states, transition[0])
        self.actions = np.append(self.actions, transition[1])
        self.rewards = np.append(self.rewards, transition[2])
        self.nextS = np.append(self.nextS, transition[3])
        self.dones = np.append(self.dones, transition[4])
        if self.algo == 'PPO':
            self.log_prob = np.append(self.log_prob, transition[5])
            
        if(self.len > self.maxlen):
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.rewards = self.rewards[1:]
            self.nextS = self.nextS[1:]
            self.dones = self.dones[1:]
            if self.algo == 'PPO':
                self.log_prob = self.log_prob[1:]
            self.len = self.maxlen

    def sample_batch(self, algo, batch_size):
        indices = np.random.randint(0, self.len, size=batch_size)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        nextS = self.nextS[indices]
        dones = self.dones[indices]
        if algo == 'PPO':
            log_prob = self.log_prob[indices]
            return [states, actions, rewards, nextS, dones, log_prob]
        
        return [states, actions, rewards, nextS, dones]

class DDPG:
    def __init__(self, env=None, episodes=100, maxlen=1000, batch_size=32, gamma=0.95):
        self.env = env
        self.name = 'DDPG'
        self.episodes = episodes
        self.buffer = Buffer(maxlen, self.name)
        self.batch_size = batch_size
        self.epochs = 0
        self.gamma = gamma
        self.tau = 0.05
        self.state = np.zeros(self.env.action_size)
        self.actor, self.target_actor = self.actors()
        self.critic, self.target_critic = self.critics()
        self.actor_optimizer = Adam(1e-4)
        self.critic_optimizer = Adam(1e-4)
        self.actor_losses = []
        self.critic_losses = []
        self.reward_each_episode = []
        self.cum_avg_rewards = []
        self.score_each_episode = []
        self.cum_avg_score = []
        self.steps = []
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        
        self.save_folder = self.env.name + '-' + self.name + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.create_save_file()
        
    def actors(self):        
        model1 = Sequential([
            Dense(16, activation=LeakyReLU()),
            Dense(32, activation=LeakyReLU()),
            Dense(64, activation=LeakyReLU()),
            Dropout(0.2),
            Dense(32, activation=LeakyReLU()),
            Dense(16, activation=LeakyReLU()),
            Dropout(0.2),
            Dense(self.env.action_size, activation='tanh')
        ])
        
        model2 = Sequential([
            Dense(16, activation=LeakyReLU()),
            Dense(32, activation=LeakyReLU()),
            Dense(64, activation=LeakyReLU()),
            Dropout(0.2),
            Dense(32, activation=LeakyReLU()),
            Dense(16, activation=LeakyReLU()),
            Dropout(0.2),
            Dense(self.env.action_size, activation='tanh')
        ])
        
        return model1, model2
        
    def critics(self):
        state_input = Input(shape=(self.env.state_size,), dtype=tf.float64)
        action_input = Input(shape=(self.env.action_size,), dtype=tf.float64)
        
        state_l1 = Dense(16, activation=LeakyReLU())(state_input)
        state_l2 = Dropout(0.2)(state_l1)
        state_output = Dense(32, activation=LeakyReLU())(state_l2)
        
        action_l1 = Dense(16, activation=LeakyReLU())(action_input)
        action_l2 = Dropout(0.2)(action_l1)
        action_output = Dense(32, activation=LeakyReLU())(action_l2)
        
        inputs = Concatenate()([state_output, action_output])
        inputs = Dense(16, activation=LeakyReLU())(inputs)
        inputs = Dense(32, activation=LeakyReLU())(inputs)
        inputs = Dropout(0.2)(inputs)
        inputs = Dense(64, activation=LeakyReLU())(inputs)
        inputs = Dense(64, activation=LeakyReLU())(inputs)
        inputs = Dropout(0.2)(inputs)
        inputs = Dense(32, activation=LeakyReLU())(inputs)
        inputs = Dense(16, activation=LeakyReLU())(inputs)
        inputs = Dropout(0.2)(inputs)
        output = Dense(1, activation='linear')(inputs)
        
        model1 = Model(inputs=[state_input, action_input], outputs=output)
        model2 = Model(inputs=[state_input, action_input], outputs=output)
        
        return model1, model2
    
    def update_target_networks(self):
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1-self.tau) * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)
        
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1-self.tau) * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)
        
    def addNoise(self, action):
        mu = 0
        sigma = 0.2
        theta = 0.15
        dt = 0.01
        x = self.state
        
        noise = theta*(mu-x)*dt + sigma*np.sqrt(dt)*np.random.normal(self.env.action_size)
        noise *= 0.1
        self.state = x + noise

        action = np.clip(action+noise, self.env.action_min, self.env.action_max)
        return action
        
    def learn(self):
        with open(self.env.name + '-' + self.name +'/env.pkl', 'wb') as file:
            pickle.dump(self.env, file)
        print('Env saved')
        for e in range(self.episodes):
            state = self.env.reset()
            r = 0
            step = 0
            done = False
            while True:
                step += 1
                action = np.clip(self.actor.predict(np.atleast_2d(state), verbose=0)[0], self.env.action_min, self.env.action_max)
                action = self.addNoise(action)
                nextS, reward, done = self.env.step(action)
                self.env.render()
                if done:
                    done = 1
                else:
                    done = 0
                self.buffer.append((state, action, reward, nextS, done))
                state = nextS
                r += reward
                if done:
                    break
                if len(self.buffer)>self.batch_size:
                    batch = random.sample(self.buffer, self.batch_size)
                    batch_states = tf.convert_to_tensor([sample[0] for sample in batch])
                    batch_actions = tf.convert_to_tensor([sample[1] for sample in batch])
                    batch_rewards = tf.convert_to_tensor([sample[2] for sample in batch])
                    batch_nextS = tf.convert_to_tensor([sample[3] for sample in batch])
                    batch_dones = tf.convert_to_tensor([sample[4] for sample in batch])
                    
                    target_actions = self.target_actor(batch_nextS)
                    target_q_values = tf.cast(batch_rewards, tf.float64) + self.gamma * tf.cast(self.target_critic([batch_nextS, target_actions]), tf.float64) * (tf.cast(tf.constant(1.0),tf.float64)-tf.cast(batch_dones, tf.float64))
                    
                    with tf.GradientTape() as tape:
                        q_values = self.critic([batch_states, batch_actions])
                        critic_loss = tf.reduce_mean(tf.square(target_q_values - tf.cast(q_values, tf.float64)))
                        
                    critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                    
                    with tf.GradientTape() as tape:
                        actions = self.actor(batch_states, training=True)
                        q_values = self.critic([batch_states, actions])
                        actor_loss = -tf.reduce_mean(q_values)
                        
                    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                    self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                    
                    if self.epochs%10==0:
                        self.update_target_networks()
                        
                    if self.epochs%100==0:
                        print('Epoch:{} Actor Loss: {} Critic Loss:{}'.format(self.epochs, actor_loss, critic_loss))
                        print('Saving model')
                        self.actor.save(self.env.name + '-' + self.name +'/actor.keras')
                        self.critic.save(self.env.name + '-' + self.name +'/critic.keras')
                        self.target_actor.save(self.env.name + '-' + self.name +'/target_actor.keras')
                        self.target_critic.save(self.env.name + '-' + self.name +'/target_critic.keras')
                        self.actor_losses.append(actor_loss)
                        self.critic_losses.append(critic_loss)
                   
                    self.epochs += 1
                         
            self.reward_each_episode.append(r)
            self.steps.append(step)
            self.plot_rewards()
        self.cum_avg_rewards = [np.mean(self.reward_each_episode[:i]) for i in range(1,len(self.reward_each_episode))]
    
    def score(self):
        for e in range(self.episodes):
            self.env.reset()
            state = 0
            r = 0
            done = False
            while not done:
                action = np.clip(self.actor.predict(np.atleast_2d(state), verbose=0)[0], self.env.action_min, self.env.action_max)
                nextS, reward, done = self.env.step(action)
                self.env.render()
                state = nextS
                r += reward
            self.score_each_episode.append(r)
            self.plot_rewards()
        self.cum_avg_score = [np.mean(self.score_each_episode[:i]) for i in range(1,len(self.score_each_episode))]
    
    def plot_rewards(self):
        self.ax.clear()
        
        self.ax.plot(self.reward_each_episode)
        
        self.ax.grid(False)
        self.ax.set_xlabel('Episodes')
        self.ax.set_ylabel('Reward')
        self.ax.set_title('Drone Environment')
        plt.pause(0.0001)
        
    def plot_score(self):
        self.ax.clear()
        
        self.ax.plot(self.score_each_episode)
        
        self.ax.grid(False)
        self.ax.set_xlabel('Episodes')
        self.ax.set_ylabel('Reward')
        self.ax.set_title('Drone Environment')
        plt.pause(0.0001)
    


class PPO():
    def __init__(self, env, episodes=100, rollout_steps=5000, score_steps=1000, num_workers=5, num_players=5, maxlen=10000, batch_size=256, gamma=0.99):
        self.name = 'PPO'
        self.env = env
        self.num_workers = num_workers
        self.num_players = num_players
        self.buffer = deque(maxlen=maxlen)
        self.episodes = episodes
        self.ep = 0
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.score_steps = score_steps
        self.gamma = gamma
        self.clip_ratio = 0.2
        self.entropy_coeff = 0.01
        self.epochs = 0
        self.epochs_per_episode = 100
        self.state = np.zeros(self.env.action_size)
        self.actor_losses = []
        self.critic_losses = []
        self.episode_rewards = []
        self.reward_each_episode = []
        self.cum_avg_rewards = []
        self.score_each_episode = []
        self.cum_avg_score = []
        self.steps = []
        
        self.save_folder = self.env.name + '-' + self.name + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.create_save_file()
        
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = Adam(1e-4)
        self.critic_optimizer = Adam(1e-5)
        self.actor.compile()
        self.critic.compile()
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
        self.info = {}
        self.info['Episode'] = []
        
        for i in range(self.num_workers):
            self.info[f'Rollout Steps (W{i+1})'] = []
            self.info[f'Reward (W{i+1})'] = []
            self.info[f'Done (W{i+1})'] = []
            
        self.info['Avg Reward'] = []
        self.info['Actor Loss'] = []
        self.info['Critic Loss'] = []
        
        for i in range(self.num_players):
            self.info[f'Steps (P{i+1})'] = []
            self.info[f'Score (P{i+1})'] = []
            self.info[f'Done (P{i+1})'] = []
            
        self.info['Avg Score'] = []
        
        pd.DataFrame(self.info).to_csv(self.save_folder + '/info.csv', index=False)
        
    def save_info(self, episode, rollout_steps, rewards, rollout_dones, actor_loss, critic_loss, steps, scores, score_dones):
        new_info = []
        
        self.info['Episode'].append(episode)
        
        for i in range(self.num_workers):
            self.info[f'Rollout Steps (W{i+1})'].append(rollout_steps[i])
            self.info[f'Reward (W{i+1})'].append(rewards[i])
            self.info[f'Done (W{i+1})'].append(rollout_dones[i])
            
        self.info['Avg Reward'].append(np.mean(rewards))
        self.info['Actor Loss'].append(actor_loss)
        self.info['Critic Loss'].append(critic_loss)
        
        for i in range(self.num_players):
            self.info[f'Steps (P{i+1})'].append(steps[i])
            self.info[f'Score (P{i+1})'].append(scores[i])
            self.info[f'Done (P{i+1})'].append(score_dones[i])
            
        self.info['Avg Score'].append(np.mean(scores))
        
        pd.DataFrame(self.info).to_csv(self.save_folder + '/info.csv', index=False)
        np.save(self.save_folder + '/buffer.npy', np.array(self.buffer))
        
    def save_info_async(self, episode, rollout_steps, rewards, rollout_dones, actor_loss, critic_loss, steps, scores, score_dones):
        thread = threading.Thread(target=self.save_info, args=(episode, rollout_steps, rewards, rollout_dones, actor_loss, critic_loss, steps, scores, score_dones))
        thread.start()
    
    def build_actor(self):
        state_input = Input(shape=(self.env.state_size,), dtype=tf.float32)
        x = Dense(16, activation='relu')(state_input)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
        
        mean_output = Dense(self.env.action_size, activation='tanh')(x)
        std_output = Dense(self.env.action_size, activation='softplus')(x)
        
        model = Model(inputs=state_input, outputs=[mean_output, std_output])
        
        return model
    
    def build_critic(self):
        state_input = Input(shape=(self.env.state_size,), dtype=tf.float32)
        x = Dense(16, activation='relu')(state_input)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
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
    
    def compute_advantages_and_targets(self, rewards, dones, q_values, next_q_values):
        batch_size = tf.shape(rewards)[0]
        advantages = tf.expand_dims(tf.ones_like(rewards, dtype=tf.float32), axis=-1)
        targets = tf.ones_like(rewards, dtype=tf.float32)
        running_advantage = tf.expand_dims([0.0], axis=-1)

        for t in tf.range(batch_size-1, -1, -1):
            r_t = tf.gather(rewards, t)
            done_t = tf.gather(dones, t)
            next_q_t = tf.gather(next_q_values, t)
            q_t = tf.gather(q_values, t)
            delta = r_t + self.gamma * tf.cast(1.0 - done_t, tf.float32) * next_q_t  - q_t
            running_advantage = tf.cast(delta, tf.float32) + self.gamma * 0.95 * tf.cast(1.0 - done_t, tf.float32) * tf.cast(running_advantage, tf.float32)
            idx = tf.reshape(t, (1,1))
            advantages = tf.tensor_scatter_nd_update(advantages, idx, running_advantage)

        targets = rewards + self.gamma * tf.cast(1.0 - dones, tf.float32) * tf.cast(next_q_values, tf.float32)

        return advantages, targets
      
    def worker(self, env, rollout_steps, actor):
        state = env.reset()
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        rollouts = []
        steps = 0
        r = 0
        for _ in range(rollout_steps):
            steps += 1
            mean, std = actor(state_tensor, training=False)
            dist = tfp.distributions.Normal(mean[0], std[0])
            action = tf.clip_by_value(dist.sample(), env.action_min, env.action_max)
            log_prob = tf.reduce_sum(dist.log_prob(action)).numpy()
            nextS, reward, done = env.step(action.numpy())
            r += reward
            state_tensor = tf.convert_to_tensor(nextS[np.newaxis, :], dtype=tf.float32)
            if done:
                done = 1
            else:
                done = 0
            rollouts.append((state, action, reward, nextS, done, log_prob))
            if done:
                break
        return rollouts, r, steps, bool(done)
        
    def collect_rollouts(self):
        rollouts = []
        with Pool(self.num_workers) as pool:
            results = pool.starmap(self.worker, [(copy.deepcopy(self.env), self.rollout_steps // self.num_workers, self.actor) for _ in range(self.num_workers)])
        
        rollouts, rewards, steps, dones = zip(*results)
        rollouts = [item for sublist in rollouts for item in sublist]
        self.buffer.extend(rollouts)
        
        return steps, rewards, dones
    
    def train_models(self):
        for _ in range(self.epochs_per_episode):
            self.epochs += 1
            batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
            batch_states = np.array([sample[0] for sample in batch], dtype=np.float32)
            batch_actions = np.array([sample[1] for sample in batch], dtype=np.float32)
            batch_rewards = np.array([sample[2] for sample in batch], dtype=np.float32)
            batch_nextS = np.array([sample[3] for sample in batch], dtype=np.float32)
            batch_dones = np.array([sample[4] for sample in batch], dtype=np.float32)
            batch_log_probs = np.array([sample[5] for sample in batch], dtype=np.float32)
            
            batch = tf.data.Dataset.from_tensor_slices((batch_states, batch_actions, batch_rewards, batch_nextS, batch_dones, batch_log_probs))
            batch = batch.shuffle(1000).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            
            current_q_values = self.critic(batch_states, training=False)
            next_q_values = self.critic(batch_nextS, training=False)
            
            batch_advantages, batch_targets = self.compute_advantages_and_targets(batch_rewards, batch_dones, current_q_values, next_q_values)
            
            actor_loss = tf.constant(0.0)
            critic_loss = tf.constant(0.0)
            for batch_states, batch_actions, batch_rewards, batch_nextS, batch_dones, batch_log_probs in batch:
                with tf.GradientTape(persistent=True) as tape:
                    mean, std = self.actor(batch_states, training=True)
                    log_probs = -0.5 * (((batch_actions - mean) / (std + 1e-8)) ** 2 + 2 * tf.math.log(std + 1e-8) + tf.math.log(2 * np.pi))
                    log_probs = tf.reduce_sum(log_probs, axis=-1)

                    ratio = tf.exp(log_probs - batch_log_probs)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    entropy = -tf.reduce_sum(log_probs, axis=-1)
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * batch_advantages, clipped_ratio * batch_advantages)) - self.entropy_coeff * tf.reduce_mean(entropy)
                    
                    values = self.critic(batch_states, training=True)
                    critic_loss = tf.reduce_mean((batch_targets - values) ** 2)

                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        self.save_models_async()
        
        return actor_loss.numpy(), critic_loss.numpy()
        
    
    def learn(self):
        with open(self.save_folder +'/env.pkl', 'wb') as file:
            pickle.dump(self.env, file)
        print('Env saved')
        self.log('Env saved')
        self.save_models()
        
        for self.ep in range(self.episodes):
            print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Episode {self.ep + 1}/{self.episodes}:')
            self.log(f'Episode {self.ep + 1}/{self.episodes}:')
            
            print('Executing rollouts...')
            self.log('Executing rollouts...')
            t_steps, t_rewards, t_dones = self.collect_rollouts()
            
            print('Training models...')
            self.log('Training models...')
            a_loss, c_loss = self.train_models()
            
            print('Playing episode...')
            self.log('Playing episode...')
            s_steps, s_rewards, s_dones = self.score()
            
            print(f'Reward: {np.mean(self.reward_each_episode[-self.num_players:]):.2f}')
            self.log(f'Reward: {np.mean(self.reward_each_episode[-self.num_players:]):.2f}')
            self.save_info_async(self.ep + 1, t_steps, t_rewards, t_dones, a_loss, c_loss, s_steps, s_rewards, s_dones)
        
    def plot_rewards(self):
        self.ax.clear()
        self.ax.plot(self.cum_avg_rewards)
        self.ax.grid(False)
        self.ax.set_xlabel('Episodes')
        self.ax.set_ylabel('Reward')
        self.ax.set_title(self.env.name + '-' + self.name)
        plt.savefig(self.save_folder + '/rewards.jpg')
        
    def plot_async(self):
        thread = threading.Thread(target=self.plot_rewards)
        thread.start()
        
    def player(self, env, actor):
        state = env.reset()
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        r = 0
        steps = 0
        for _ in range(self.score_steps):
            steps += 1
            mean, std = actor(state_tensor, training=False)
            dist = tfp.distributions.Normal(mean[0], std[0])
            action = tf.clip_by_value(dist.sample(), env.action_min, env.action_max)
            log_prob = tf.reduce_sum(dist.log_prob(action)).numpy()
            nextS, reward, done = env.step(action)
            state_tensor = tf.convert_to_tensor(nextS[np.newaxis, :], dtype=tf.float32)
            r += reward
            if done:
                break
        return r, steps, bool(done)
        
    def score(self):
        with Pool(self.num_workers) as pool:
            results = pool.starmap(self.player, [(copy.deepcopy(self.env), self.actor) for _ in range(self.num_players)])
        
        rewards, steps, dones = zip(*results)  
        self.reward_each_episode.extend(rewards)
        self.steps.extend(steps)

        self.cum_avg_rewards.append(np.mean(self.reward_each_episode))
        self.plot_async()
        
        return steps, rewards, dones
            
            
            
            
if __name__ == "__main__":
    pass