from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Envs.Country import CountryEnvBox



env = CountryEnvBox()
vec_env = make_vec_env(CountryEnvBox, n_envs=5)
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000000)
model.save('CountryEnvSB3PPO')

del model

env = CountryEnvBox()
max_steps = 100
model = PPO.load('CountryEnvSB3PPO')
state, _ = env.reset()
# r = 0
steps = 0
env.render()

while steps < max_steps:
    action, _ = model.predict(state)
    state, reward, done, _, _ = env.step(action)
    env.render()
    # r += reward
    steps += 1
    if done:
        break