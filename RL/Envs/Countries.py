import numpy as np
import pygame
import gym
from gym import spaces

# Constants
grid_size = 4  # 4x4 battlefield
num_countries = 16  # Each country gets 1 tile
infantry_power = 10
tank_power = 50  # 1 tank = 5 infantry

# Economy
resources = {
    "gold": 100,
    "oil": 50,
    "metal": 50,
    "processed_metal": 0,  # Used for making tanks
    "consumer_goods": 0  # Maintains population sentiment
}

class MilitaryEnv(gym.Env):
    def __init__(self):
        super(MilitaryEnv, self).__init__()
        self.grid = np.zeros((grid_size, grid_size))  # Empty battlefield
        self.countries = {i: {"infantry": 5, "tanks": 1, "factories": {"consumer": 1, "military": 1}, "resources": resources.copy()} for i in range(num_countries)}
        self.action_space = spaces.Discrete(4)  # Move, attack, build, produce
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        
        # Process economy (factories produce goods per step)
        for country in self.countries.values():
            if country["factories"]["military"] > 0:
                country["resources"]["processed_metal"] += country["factories"]["military"]  # 1 processed metal per step
            if country["factories"]["consumer"] > 0:
                country["resources"]["consumer_goods"] += country["factories"]["consumer"]  # 1 consumer good per step
            
            # Population sentiment check (if consumer goods = 0, sentiment drops)
            if country["resources"]["consumer_goods"] == 0:
                reward -= 5  # Population discontent
        
        return self.grid, reward, done, {}

    def reset(self):
        self.grid.fill(0)
        self.countries = {i: {"infantry": 5, "tanks": 1, "factories": {"consumer": 1, "military": 1}, "resources": resources.copy()} for i in range(num_countries)}
        return self.grid

    def render(self):
        pygame.init()
        screen = pygame.display.set_mode((400, 400))
        screen.fill((0, 0, 0))
        tile_size = 100
        
        for i in range(grid_size):
            for j in range(grid_size):
                country_id = i * grid_size + j
                pygame.draw.rect(screen, (100, 100, 255), (j * tile_size, i * tile_size, tile_size, tile_size))
                pygame.draw.rect(screen, (0, 0, 0), (j * tile_size, i * tile_size, tile_size, tile_size), 2)
        
        pygame.display.flip()

# Example usage
env = MilitaryEnv()
env.reset()
for _ in range(10):
    env.step(env.action_space.sample())
    env.render()
