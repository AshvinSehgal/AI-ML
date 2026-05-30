import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CountryEnv:
    def __init__(self, max_workers=1000):
        self.max_workers = max_workers
        self.max_assignments = 5
        self.size = 100
        self.state_size = 5 # (Oil, Gold, Metal, Food, Money, Population)
        self.action_size = 4 # (Oil, Gold, Metal, Food)
        self.resources = np.array(np.random.rand(6) * [100, 50, 50, 80, 100, 1000], dtype=np.float32) # (Oil, Gold, Metal, Food, Money, Population)
        self.inventory = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.buildings = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Wells, Gold Mines, Metal Mines, Farms)
        self.buildings_cost = np.array([50, 100, 75, 25], dtype=np.float32) # (Oil Wells, Gold Mines, Metal Mines, Farms)
        self.buildings_capacity = np.array([20, 25, 20, 10], dtype=np.float32) # (Oil Wells, Gold Mines, Metal Mines, Farms)
        self.workers = np.array([0, 0, 0, 0, self.resources[5]], dtype=np.float32) # (Oil Workers, Gold Workers, Metal Workers, Farmers, Available Workers)
        self.tech_level = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Tech, Gold Tech, Metal Tech, Food Tech)
        self.prices = np.array([5, 10, 3, 1], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.costs = np.array([1, 2, 0.5, 0.2], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.state = self.get_state()
        
    def reset(self):
        self.resources = np.array(np.random.rand(6) * [100, 50, 50, 80, 100, 1000], dtype=np.float32) # (Oil, Gold, Metal, Food, Money, Population)
        self.inventory = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.buildings = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Wells, Gold Mines, Metal Mines, Farms)
        self.workers = np.array([0, 0, 0, 0, 0], dtype=np.float32) # (Oil Workers, Gold Workers, Metal Workers, Farmers, Available Workers)
        self.tech_level = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Tech, Gold Tech, Metal Tech, Food Tech)
        self.state = self.get_state()
        return self.state
    
    def get_state(self):
        return np.concatenate([self.resources, self.inventory, self.buildings, self.workers, self.tech_level])
    
    def apply_worker_assignments(self, assignment_vector):
        M = np.zeros((5,5), dtype=np.int32)
        idx = 0
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                M[i, j] = assignment_vector[idx]
                idx += 1
        
        outgoing = M.sum(axis=1)
        scale_rows = np.ones(5, dtype=np.float32)
        mask = outgoing > self.workers
        if mask.any():
            scale_rows[mask] = (self.workers[mask] / np.maximum(outgoing[mask], 1)).astype(np.float32)
            M = (M.astype(np.float32) * scale_rows[:, np.newaxis]).astype(np.int32)
            outgoing = M.sum(axis=1)
        
        building_capacity = (self.buildings * self.buildings_capacity).astype(np.int32) - self.workers[:-1]
        building_capacity = np.maximum(building_capacity, 0)
        capacity_left = np.append(building_capacity,np.array([self.resources[5] - self.workers[4]]))
        capacity_left = np.maximum(capacity_left, 0)
        incoming = M.sum(axis=0)
        mask = incoming > capacity_left
        if mask.any():
            scale_cols = np.minimum(1.0, capacity_left / np.maximum(incoming, 1))
            M = (M.astype(np.float32) * scale_cols[np.newaxis, :]).astype(np.int32)
            incoming = M.sum(axis=0)
        
        self.workers = self.workers - outgoing + incoming
        self.workers = np.maximum(self.workers, 0)
        total_pop = int(round(self.resources[5]))
        current_total = int(self.workers.sum())
        if current_total > total_pop:
            excess = current_total - total_pop
            reduce_from_available = min(excess, self.workers[4])
            self.workers[4] -= reduce_from_available
            excess -= reduce_from_available
            if excess > 0:
                sector_sum = max(1, self.workers[:4].sum())
                reduction = ((self.workers[:4].astype(np.float32) / sector_sum) * excess).astype(np.int32)
                self.workers[:4] = np.maximum(self.workers[:4] - reduction, 0)
    
    def step(self, action):
        tech_mult = 1 + self.tech_level
        expected_production = self.workers[:4].astype(np.float32) * tech_mult
        produced = np.minimum(expected_production, self.resources[:4])
        produced = np.floor(produced)
        self.resources[:4] -= produced
        self.inventory += produced
        
        assignments = action[:20].astype(np.int32)
        new_buildings = action[20:24].astype(np.int32)
        inventory_sell = action[24:28].astype(np.int32)
        
        self.apply_worker_assignments(assignments)
            
        self.buildings += new_buildings
        self.resources[4] -= np.sum(new_buildings * self.buildings_cost)
        if self.resources[4] < 0:
            penalty = -10 * abs(self.resources[4])
            self.resources[4] = 0
            self.state = self.get_state()
            return self.state, penalty, False
        
        oil, gold, metal, food = self.inventory
        d_oil, d_gold, d_metal, d_food = inventory_sell
        if d_oil < 0: # Sell Oil
            d_oil = np.abs(max(d_oil, -self.inventory[0]))
            oil -= d_oil
            self.resources[4] += d_oil * self.prices[0] 
        if d_gold < 0: # Sell Gold
            d_gold = np.abs(max(d_gold, -self.inventory[1]))
            gold -= d_gold
            self.resources[4] += d_gold * self.prices[1]
        if d_metal < 0: # Sell Metal
            d_metal = np.abs(max(d_metal, -self.inventory[2]))
            metal -= d_metal
            self.resources[4] += d_metal * self.prices[2]
        if d_food < 0: # Sell Food
            d_food = np.abs(max(d_food, self.inventory[3]))
            food -= d_food
            self.resources[4] += d_food * self.prices[3]
        
        self.inventory = np.array([oil, gold, metal, food], dtype=np.float32)
        self.state = self.get_state()
        if self.resources[4] < 0:
            return self.state, -100, True
        reward = np.sum(self.inventory * self.prices) + self.resources[4]
        return self.state, reward, False
    
    def render(self):
        oil, gold, metal, food = self.inventory
        money = self.resources[4]
        print(f"Oil: {oil}, Gold: {gold}, Metal: {metal}, Food: {food}, Money: {money}")
        
    def seed(self, seed=None):
        np.random.seed(42)
        
class CountryEnvBox(gym.Env):
    def __init__(self):
        super(CountryEnvBox,self).__init__()
        self.t = 0
        self.max_steps = 10000
        self.init_resources = np.array(np.array(np.random.rand(6) * [100, 50, 50, 80, 100, 1000],dtype=np.int32), dtype=np.float32) # (Oil, Gold, Metal, Food, Money, Population)
        self.resources = self.init_resources # (Oil, Gold, Metal, Food, Money, Population)
        self.inventory = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.buildings = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Wells, Gold Mines, Metal Mines, Farms)
        self.buildings_cost = np.array([50, 100, 75, 25], dtype=np.float32) # (Oil Wells, Gold Mines, Metal Mines, Farms)
        self.buildings_capacity = np.array([20, 25, 20, 10], dtype=np.float32) # (Oil Wells, Gold Mines, Metal Mines, Farms)
        self.workers = np.array([0, 0, 0, 0, self.init_resources[5]], dtype=np.float32) # (Oil Workers, Gold Workers, Metal Workers, Farmers, Available Workers)
        self.tech_level = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Tech, Gold Tech, Metal Tech, Food Tech)
        self.prices = np.array([5, 10, 3, 1], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.costs = np.array([1, 2, 0.5, 0.2], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.materials = self.init_resources[:4]
        self.money = self.init_resources[4]
        self.population = self.init_resources[5]
        self.state = self.get_state()
        self.observation_space = gym.spaces.Box(
            low=np.zeros(23, dtype=np.float32),
            high=np.concatenate([np.append(self.materials, [self.money, self.population]), self.materials, np.inf*np.ones(4), self.population*np.ones(5), np.inf*np.ones(4)], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.concatenate([np.zeros(20), np.zeros(4), -self.materials], dtype=np.float32),
            high=np.concatenate([self.population*np.ones(20), 1000*np.ones(4), self.materials], dtype=np.float32),
            dtype=np.float32,
        )
        
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(int(seed))
        self.t = 0
        self.resources = self.init_resources # (Oil, Gold, Metal, Food, Money, Population)
        self.inventory = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil, Gold, Metal, Food)
        self.buildings = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Wells, Gold Mines, Metal Mines, Farms)
        self.workers = np.array([0, 0, 0, 0, self.init_resources[5]], dtype=np.float32) # (Oil Workers, Gold Workers, Metal Workers, Farmers, Available Workers)
        self.tech_level = np.array([0, 0, 0, 0], dtype=np.float32) # (Oil Tech, Gold Tech, Metal Tech, Food Tech)
        self.state = self.get_state()
        return self.state, {}

    def get_state(self):
        return np.concatenate([self.resources, self.inventory, self.buildings, self.workers, self.tech_level])

    def apply_worker_assignments(self, assignment_vector):
        M = np.zeros((5,5), dtype=np.int32)
        idx = 0
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                M[i, j] = assignment_vector[idx]
                idx += 1
        
        outgoing = M.sum(axis=1)
        scale_rows = np.ones(5, dtype=np.float32)
        mask = outgoing > self.workers
        if np.any(mask):
            scale_rows[mask] = (self.workers[mask] / np.maximum(outgoing[mask], 1)).astype(np.float32)
            M = (M.astype(np.float32) * scale_rows[:, np.newaxis]).astype(np.int32)
            outgoing = M.sum(axis=1)
        
        building_capacity = (self.buildings * self.buildings_capacity).astype(np.int32) - self.workers[:-1]
        building_capacity = np.maximum(building_capacity, 0)
        capacity_left = np.append(building_capacity,self.resources[5] - self.workers[4])
        capacity_left = np.maximum(capacity_left, 0)
        incoming = M.sum(axis=0)
        mask = incoming > capacity_left
        if np.any(mask):
            scale_cols = np.minimum(1.0, capacity_left / np.maximum(incoming, 1))
            M = (M.astype(np.float32) * scale_cols[np.newaxis, :]).astype(np.int32)
            incoming = M.sum(axis=0)
        
        incoming = np.maximum(0, incoming)
        outgoing = np.maximum(0, outgoing)
        self.workers = self.workers - outgoing + incoming
        self.workers = np.maximum(self.workers, 0)
        current_total = int(self.workers.sum())
        if current_total > self.population:
            excess = current_total - self.population
            reduce_from_available = min(excess, self.workers[4])
            self.workers[4] -= reduce_from_available
            excess -= reduce_from_available
            if excess > 0:
                sector_sum = max(1, self.workers[:4].sum())
                reduction = ((self.workers[:4].astype(np.float32) / sector_sum) * excess).astype(np.int32)
                self.workers[:4] = np.maximum(self.workers[:4] - reduction, 0)
        # print(f'Incoming: {incoming}, Outgoing: {outgoing}\n')

    def step(self, action):
        self.t += 1
        reward = 0
        net_before = self.resources[4] + np.sum(self.inventory * self.prices)
        assignments = action[:20].astype(np.int32)
        new_buildings = action[20:24].astype(np.int32)
        new_buildings = np.maximum(0, new_buildings)
        inventory_sell = action[24:28].astype(np.int32)
        
        self.apply_worker_assignments(assignments)
            
        cost = np.sum(new_buildings * self.buildings_cost)
        if cost > self.resources[4]:
            penalty = self.resources[4] - cost
            self.resources[4] = 0
            reward += penalty
        else:
            self.buildings += new_buildings
            self.resources[4] -= cost
        
        oil, gold, metal, food = self.inventory
        d_oil, d_gold, d_metal, d_food = inventory_sell
        if d_oil < 0: # Sell Oil
            d_oil = np.abs(max(int(d_oil), int(-self.inventory[0])))
            oil -= d_oil
            self.resources[4] += d_oil * self.prices[0] 
        if d_gold < 0: # Sell Gold
            d_gold = np.abs(max(int(d_gold), int(-self.inventory[1])))
            gold -= d_gold
            self.resources[4] += d_gold * self.prices[1]
        if d_metal < 0: # Sell Metal
            d_metal = np.abs(max(int(d_metal), int(-self.inventory[2])))
            metal -= d_metal
            self.resources[4] += d_metal * self.prices[2]
        if d_food < 0: # Sell Food
            d_food = np.abs(max(int(d_food), int(-self.inventory[3])))
            food -= d_food
            self.resources[4] += d_food * self.prices[3]
        
        expected_production = self.workers[:4].astype(np.float32) * (1 + self.tech_level)
        produced = np.minimum(expected_production, self.resources[:4]).astype(np.float32)
        self.inventory = np.array([oil, gold, metal, food], dtype=np.float32) + produced
        reward += self.resources[4] + np.sum(self.inventory * self.prices) - net_before
        self.state = self.get_state()
        return self.state, reward, False, self.t >= self.max_steps, {}

    def render(self):
        oil, gold, metal, food = self.inventory
        money = self.resources[4]
        print(f"Oil: {oil}, Gold: {gold}, Metal: {metal}, Food: {food}, Money: {money}")
        
    def seed(self, seed=None):
        np.random.seed(seed)