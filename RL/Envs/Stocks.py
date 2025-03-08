import numpy as np
import pandas as pd
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import yfinance as yf

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df # List of stock prices
        self.current_step = 0
        self.balance = 1000000  # Starting cash
        self.stocks = df.shape[1]  # Number of stocks
        self.shares_held = np.array([0] * self.stocks)  # Shares held for each stock
        self.total_value = self.balance
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.stocks,), dtype=np.float32)
    
    def reset(self, seed=None):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = [0] * self.stocks
        self.total_value = self.balance
        vals = self.df.iloc[self.current_step].values
        return np.array([float(p) for p in vals]), {}
    
    def seed(self, seed=None):
        return np.random.seed(100)
    
    def step(self, action):
        # -1 = Sell All, 0 means hold, 1 = Buy Max
        current_prices = self.df.iloc[self.current_step].values
        current_prices = np.array([float(p) for p in current_prices])
        for i in range(self.stocks):
            if action[i] > 0:
                max_shares = self.balance // current_prices[i]
                num_shares = self.shares_held[i] + (max_shares * action[i])
                cost = (num_shares - self.shares_held[i]) * current_prices[i]
                self.shares_held[i] = num_shares
                self.balance -= cost
            elif action[i] < 0:
                num_shares = int((1 - abs(action[i])) * self.shares_held[i])
                earning = (self.shares_held[i] - num_shares) * current_prices[i]
                self.shares_held[i] = num_shares
                self.balance += earning
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        self.total_value = self.balance + np.sum(self.shares_held * current_prices)
        reward = self.total_value - 10000
        vals = self.df.iloc[self.current_step].values
        return np.array([float(p) for p in vals]), reward, done, {}, {}

tickers = ['AAL', 'AAPL', 'ABT', 'ADBE', 'AMZN', 'BAC', 'BLK', 'CL', 'COKE', 'CRM', 'DPZ', 'FOX', 'H', '^IXIC', 'JPM', 'LOGI', 'MSCI', 'NFLX', 'NVDA', 'RACE', 'SONY', 'TEAM', 'UAL', 'V', 'WMT', 'ZM']

for ticker in tickers:
    stock = yf.download(ticker, period='5y', interval='1d').to_csv(f'{ticker}_stock.csv')

# Load stock data
df = pd.DataFrame({
    ticker: pd.read_csv(f'{ticker}_stock.csv')['Close'].values[2:] for ticker in tickers
})

for ticker in tickers:
    os.remove(f'{ticker}_stock.csv')

# Train PPO agent
vec_env = make_vec_env(lambda: StockTradingEnv(df), n_envs=10)
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000000)
model.save('Stocks-PPO')

del model

# Test agent
model = PPO.load('Stocks-PPO')
env = StockTradingEnv(df)
done = False
state, _ = env.reset()
while not done:
    action, _ = model.predict(state)
    state, reward, done, _, _ = env.step(action)
    print(f"Step: {env.current_step}, Current Value: {env.total_value}")