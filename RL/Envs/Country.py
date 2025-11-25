import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CountryEnv:
    def __init__(self):
        self.size = 100
        self.state_size = 5 # (Oil, Gold, Metal, Food, Money, Population)
        self.action_size = 4 # (Oil, Gold, Metal, Food)
        self.resources = np.array(np.random.rand() * [100, 50, 50, 80, 100, 1000], dtype=np.float32) # (Oil, Gold, Metal, Food, Money, Population)
        self.inventory = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.buildings = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Wells, Gold Mines, Metal Mines, Farms)
        self.workers = np.array([0, 0, 0, 0, self.resources[5]], dtype=np.float32) # (Oil Workers, Gold Workers, Metal Workers, Farmers, Available Workers)
        self.tech_level = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Tech, Gold Tech, Metal Tech, Food Tech)
        self.prices = np.array([5, 10, 3, 1], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.costs = np.array([1, 2, 0.5, 0.2], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.state = np.concatenate([self.resources, self.inventory, self.buildings, self.workers, self.tech_level])
        
    def reset(self):
        self.resources = np.array(np.random.rand() * [100, 50, 50, 80, 100, 1000], dtype=np.float32) # (Oil, Gold, Metal, Food, Money, Population)
        self.inventory = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.buildings = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Wells, Gold Mines, Metal Mines, Farms)
        self.workers = np.array([0, 0, 0, 0, 0], dtype=np.float32) # (Oil Workers, Gold Workers, Metal Workers, Farmers, Available Workers)
        self.tech_level = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Tech, Gold Tech, Metal Tech, Food Tech)
        self.state = np.concatenate([self.resources, self.inventory, self.buildings, self.workers, self.tech_level])
        return self.state
    
    def step(self, action):
        worker_count = int(action[:20])
        from_resource = np.arange(5)
        to_resource = np.arange(5)
        buildings_count = int(action[20:40])
        if worker_count > self.workers[4]: # Not enough available workers
            worker_count = self.workers[4]
            self.workers[4] -= worker_count
            self.workers[from_resource] += worker_count
            self.workers[to_resource] += worker_count
        
        new_buildings = action[2]
        oil, gold, money = self.state
        d_oil, d_gold = action
        d_money = 0
        
        if d_oil > 0: # Mine Oil
            oil += d_oil
            d_money -= d_oil * self.cost['oil']
        else: # Sell Oil
            oil += d_oil
            d_money += d_oil * self.prices['oil']
            
        if d_gold > 0: # Mine Gold
            gold += d_gold
            d_money -= d_gold * self.cost['gold']
        else: # Sell Gold
            gold += d_gold
            d_money += d_gold * self.prices['gold']
        
        money += d_money
        self.state = np.array([oil, gold, money], dtype=np.float32)
        reward = d_money
        return self.state, reward, False
    
    def render(self):
        oil, gold, money = self.state
        print(f"Oil: {oil}, Gold: {gold}, Total Money: {money}")
        
    def seed(self, seed=None):
        np.random.seed(42)
        
class CountryEnvBox(gym.Env):
    def __init__(self):
        super(CountryEnvBox, self).__init__()
        self.state_size = 3
        self.action_size = 4