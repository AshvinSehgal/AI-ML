from Envs.Govt import MultiCountryEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

vec_env = make_vec_env(lambda: Monitor(MultiCountryEnv(num_countries=10)), n_envs=5)
model = PPO("MlpPolicy", vec_env, learning_rate=0.0003, n_steps=2048, batch_size=2048, verbose=1)
model.learn(total_timesteps=1000000)
model.save('MultiGovtPPO')

# del model

model = PPO.load('GovtPPO')

env = MultiCountryEnv(num_countries=10)

obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info, _ = env.step(action)
    env.render()
    if done:
        break
