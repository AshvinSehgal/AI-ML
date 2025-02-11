import numpy as np
import matplotlib.pyplot as plt

class DroneEnvDiscrete:
    def __init__(self, rows=10, cols=10):
        self.nStates = rows * cols
        self.nActions = 4
        self.rows = rows
        self.cols = cols
        self.totalCells = rows * cols
        self.dronePos = 0
        self.SAMs = int(np.sqrt(self.totalCells)) # No. of SAM systems
        self.pos = np.random.choice(np.arange(1, self.totalCells), self.SAMs+1, replace=False)
        self.goalPos = self.pos[-1] # Goal
        self.SAMPos = self.pos[:self.SAMs] 
        self.missiles = [False for _ in range(self.SAMs)] # Missiles launched or not
        self.missilesPos = self.SAMPos.copy()
        self.range = 2 # Detection range
    
    def showEnv(self):
        grid = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
        grid[int(self.dronePos / self.cols)][self.dronePos % self.cols] = 'D'
        grid[int(self.goalPos / self.cols)][self.goalPos % self.cols] = 'G'
        for SAM in self.SAMPos:
            grid[int(SAM / self.cols)][SAM % self.cols] = 'S'
        for i in range(len(self.missiles)):
            if self.missiles[i]:
                missile = self.missilesPos[i]
                if missile == self.dronePos:
                    grid[int(missile / self.cols)][missile % self.cols] = 'X'
                else:
                    grid[int(missile / self.cols)][missile % self.cols] = 'M'
        if self.dronePos == self.goalPos:
            grid[int(self.goalPos / self.cols)][self.goalPos % self.cols] = 'Y'
        print(np.matrix(grid))
    
    def reset(self):
        self.missiles = [False for _ in range(self.SAMs)]
        self.missilesPos = self.SAMPos.copy()
        self.dronePos = 0   
    
    def updateMissilesPos(self):
        droneRow = int(self.dronePos / self.cols)
        droneCol = self.dronePos % self.cols
        for i in range(len(self.missiles)):
            missileRow = int(self.missilesPos[i] / self.cols)
            missileCol = self.missilesPos[i] % self.cols
            
            rowDiff = droneRow - missileRow
            colDiff = droneCol - missileCol
                
            if self.missiles[i]: # This missile is in-air
                directionProb  = np.random.rand() # Decide which direction to move in
                
                if rowDiff == 0:
                    directionProb = 1
                elif colDiff == 0:
                    directionProb = 0
                    
                if directionProb < 0.5: # Move in row
                    if rowDiff < 0:
                        self.missilesPos[i] -= self.cols
                    else:
                        self.missilesPos[i] += self.cols
                else: # Move in column
                    if colDiff < 0: 
                        self.missilesPos[i] -= 1
                    else:
                        self.missilesPos[i] += 1
            elif np.abs(rowDiff) + np.abs(colDiff) <= self.range: # Launch missile
                self.missiles[i] = True
        
            
    def droneIntercepted(self): # If drone has been intercepted by a missile
        for i in range(len(self.missiles)):
            if self.missiles[i] and self.missilesPos[i] == self.dronePos:
                return True
        return False
    
    def goalAchieved(self): # Drone has reached the goal
        return self.dronePos == self.goalPos
    
    def step(self, action):
        if action==0 and self.dronePos >= self.cols: # North
            self.dronePos -= self.cols
        elif action==1 and self.dronePos % self.cols > 0: # West
            self.dronePos -= 1
        elif action==2 and self.dronePos % self.cols < self.cols - 1: # East
            self.dronePos += 1
        elif action==3 and self.dronePos < self.totalCells - self.cols: # South
            self.dronePos += self.cols
        
        self.updateMissilesPos()
        
        if self.goalAchieved():
            return self.dronePos, 1000, True
        elif self.droneIntercepted():
            return self.dronePos, -100, True
        else:
            return self.dronePos, 0, False