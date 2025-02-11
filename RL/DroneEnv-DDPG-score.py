from Envs.DroneEnvContinuous import DroneEnvContinuous
from Algos.DDPG import DDPG

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

with open('env.pkl', 'rb') as file:
    env = pickle.load(file)
print('Env Loaded')

custom_objects = {
    'LeakyReLU': LeakyReLU()
}

agent = DDPG(env=env, episodes=100)
agent.actor = tf.keras.models.load_model('actor.keras', custom_objects=custom_objects)
agent.target_actor = tf.keras.models.load_model('target_actor.keras', custom_objects=custom_objects)
agent.critic = tf.keras.models.load_model('critic.keras', custom_objects=custom_objects)
agent.target_critic = tf.keras.models.load_model('target_critic.keras', custom_objects=custom_objects)
print('Models Loaded')
        
agent = DDPG(env=env, episodes=100)
agent.score()