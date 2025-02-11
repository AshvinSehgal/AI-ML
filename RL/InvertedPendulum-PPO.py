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
from Envs.DroneEnvContinuous import DroneEnvContinuous
from Algos.PPO import PPO

import numpy as np
import pickle
import tensorflow as tf
import gymnasium as gym
           
env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
agent = PPO(env, state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], max_steps=10000000, rollout_steps=2000, score_steps=2000, num_workers=5, num_players=5, maxlen=10000, batch_size=256, gamma=0.99)
agent.learn()
    
        