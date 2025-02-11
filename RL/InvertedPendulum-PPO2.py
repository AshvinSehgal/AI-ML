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
from multiprocessing import Pool
import os
import warnings
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
from Algos.PPO2 import PPO


env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
agent = PPO(env=env, state_size=4, action_size=2, max_steps=1000000, rollout_steps=5000, score_steps=5000, num_workers=5, num_players=5, maxlen=10000, batch_size=256, gamma=0.99, env_name='InvertedPendulum')
agent.learn()
    
        