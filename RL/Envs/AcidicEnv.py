import numpy as np
import matplotlib.pyplot as plt

class AcidicEnv:
    def __init__(self, x_max=10, y_max=10, z_max=10, molecules=1000):
        self.x_min = 0
        self.y_min = 0
        self.z_min = 0
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.action_min = -1
        self.action_max = 1
        self.state_size = 3
        self.action_size = 3
        self.capsulePos = np.zeros(self.state_size) # Metal capsule
        self.molecules = molecules # No. of acid molecules
        self.moleculesPos = np.array(np.random.rand(self.molecules, 3) * [self.x_max, self.y_max, self.z_max])
        self.goalPos = np.array([self.x_max, self.y_max, self.z_max])
    
    def reset(self):
        self.capsulePos = np.zeros(3)
        self.moleculesPos = np.array(np.random.rand(self.molecules, 3) * [self.x_max, self.y_max, self.z_max])
        return self.capsulePos
    
    def updateMoleculesPos(self):
        for i in range(self.molecules):
            pos_change = np.random.rand(3)
            if(np.linalg.norm(pos_change)>1):
                pos_change /= np.linalg.norm(pos_change)
            self.moleculesPos[i][0] = np.clip(self.moleculesPos[i][0]+pos_change[0], self.x_min, self.x_max)
            self.moleculesPos[i][1] = np.clip(self.moleculesPos[i][1]+pos_change[1], self.y_min, self.y_max)
            self.moleculesPos[i][2] = np.clip(self.moleculesPos[i][2]+pos_change[2], self.z_min, self.z_max)
            
    def capsuleIntercepted(self): # If capsule was dissolved by an acid molecule
        for i in range(self.molecules):
            if np.all(self.moleculesPos[i] == self.capsulePos):
                return True
        return False
    
    def goalAchieved(self): # Capsule has reached the goal
        return np.all(self.capsulePos == self.goalPos)
    
    def step(self, action):
        if(np.linalg.norm(action)>1):
            dx, dy, dz = action/np.linalg.norm(action)
        else:
            dx, dy, dz = action
        self.capsulePos[0] = np.clip(self.capsulePos[0]+dx, self.x_min, self.x_max)
        self.capsulePos[1] = np.clip(self.capsulePos[1]+dy, self.y_min, self.y_max)
        self.capsulePos[2] = np.clip(self.capsulePos[2]+dz, self.z_min, self.z_max)
        
        self.updateMoleculesPos()
        
        if self.goalAchieved():
            return self.capsulePos, 1000, True
        elif self.capsuleIntercepted():
            return self.capsulePos, -1000, True
        else:
            return self.capsulePos, 0, False