import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DroneEnvBox(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, x_max=10, y_max=10, SAMs=10):
        super(DroneEnvBox, self).__init__()
        self.name = 'DroneEnv'
        self.SAMs = 15 # No. of SAM systems
        self.x_min = 0
        self.y_min = 0
        self.x_max = x_max
        self.y_max = y_max
        self.action_min = -1
        self.action_max = 1
        self.state_size = 4
        self.action_size = 2
        
        self.observation_space = spaces.Box(
            low=np.array([self.x_min, self.y_min] * 2, dtype=np.float32),
            high=np.array([self.x_max, self.y_max] * 2, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.dronePos = np.zeros(2, dtype=np.float32)
        self.SAMPos = np.array([[10,0], [2, 0], [3, 3], [6, 4], [9, 7], [6, 0], [8, 2], [10, 4], [0, 10], [0, 2], [4, 6], [7, 9], [0, 6], [2, 8], [4, 10]], dtype=np.float32)
        self.goalPos = np.array([self.x_max, self.y_max]) # Goal
        self.range = 1 # Detection range
        # plt.ion()
        # self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig, self.ax = None, None
        
    def seed(self, seed=None):
        np.random.seed(100)
    
    def reset(self): 
        self.dronePos = np.zeros(2, dtype=np.float32)
        
        relativeGoalPos = (self.goalPos - self.dronePos)
        state = np.concatenate([self.dronePos, relativeGoalPos])
        
        return state       
            
    def droneIntercepted(self): # If drone has been intercepted by a missile
        return any(
            np.linalg.norm(self.SAMPos[i] - self.dronePos) <= self.range
            for i in range(self.SAMs)
        )
    
    def goalAchieved(self): # Drone has reached the goal
        return np.all(self.dronePos == self.goalPos)
    
    def step(self, action):
        prev_goal_distance = np.linalg.norm(self.dronePos - self.goalPos)
        
        dx, dy = action
        self.dronePos[0] = np.clip(self.dronePos[0]+dx, self.x_min, self.x_max)
        self.dronePos[1] = np.clip(self.dronePos[1]+dy, self.y_min, self.y_max)
        
        goal_distance = np.linalg.norm(self.dronePos - self.goalPos)
        
        relativeGoalPos = (self.goalPos - self.dronePos)
        state = np.concatenate([self.dronePos, relativeGoalPos])
        
        if self.droneIntercepted():
            return state, -100, True, {}
        elif self.goalAchieved():
            return state, 100, True, {}
        else:
            timePenalty = -1
            movementPenalty = prev_goal_distance - goal_distance
            return state, movementPenalty + timePenalty, False, {}
    
    def render(self, mode='human'):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))

        self.ax.clear()
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)

        self.ax.plot(self.dronePos[0], self.dronePos[1], 'bo', markersize=10, label="Drone")
        self.ax.plot(self.SAMPos[:, 0], self.SAMPos[:, 1], 'rx', markersize=8, label="SAM System")

        for pos in self.SAMPos:
            range_circle = patches.Circle((pos[0], pos[1]), self.range, color='r', alpha=0.2)
            self.ax.add_patch(range_circle)

        self.ax.plot(self.goalPos[0], self.goalPos[1], 'go', markersize=10, label="Goal")
        self.ax.grid(False)
        self.ax.legend(loc="upper right")
        self.ax.set_title("Drone Environment")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        plt.pause(0.001)
        
        def close(self):
            if self.fig:
                plt.close(self.fig)
                self.fig, self.ax = None, None