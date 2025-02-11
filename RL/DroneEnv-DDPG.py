from Envs.DroneEnvContinuous import DroneEnvContinuous
from Algos.DDPG import DDPG

import numpy as np
import pickle
import tensorflow as tf

env = DroneEnvContinuous(x_max=25, y_max=25)
agent = DDPG(env=env, episodes=100)
agent.learn()