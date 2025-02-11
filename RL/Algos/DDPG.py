import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from collections import deque

tf.config.run_functions_eagerly(True)

class DDPG:
    def __init__(self, env=None, episodes=100, maxlen=1000, batch_size=32, gamma=0.95):
        self.env = env
        self.name = 'DDPG'
        self.episodes = episodes
        self.buffer = deque(maxlen=maxlen)
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
    