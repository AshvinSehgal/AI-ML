from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

vec_env = make_vec_env(lambda: gym.make('InvertedPendulum-v5', reset_noise_scale=0.1), n_envs=5)
model = PPO("MlpPolicy", vec_env, learning_rate=0.0003, n_steps=2048, batch_size=2048, verbose=1)
model.learn(total_timesteps=10000000)
model.save('InvertedPendulumSB3PPO')

del model

# model = PPO.load('InvertedPendulumSB3PPO')

# obs = vec_env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs)
#     obs, reward, done, info, _ = vec_env.step(action)
#     vec_env.render(mode='human')
#     if done:
#         break
# vec_env.close()