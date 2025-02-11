from Envs.DroneEnvContinuous import DroneEnvContinuous
from Algos.PPO import PPO

import numpy as np
import pickle
import tensorflow as tf


env = DroneEnvContinuous()
agent = PPO(env=env, state_size=4, action_size=2, max_steps=1000000, rollout_steps=5000, score_steps=5000, num_workers=5, num_players=5, maxlen=10000, batch_size=64, gamma=0.99, env_name='DroneEnv')
agent.learn()