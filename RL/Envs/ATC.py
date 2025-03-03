import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces

class ATCEnv(gym.Env):
    """Custom ATC Environment with Runway Assignments and Visualization"""

    def __init__(self, num_aircraft=5):
        super(ATCEnv, self).__init__()
        self.num_aircraft = num_aircraft
        self.airspace_size = 100  # 100x100 NM grid
        self.altitude_levels = [30000, 32000, 34000]
        self.runways = np.array([[10, 10], [90, 90]])  # Fixed runway locations

        # Observation: x, y, z, vx, vy, vz, heading, runway_x, runway_y
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 30000, -600, -600, -100, 0, 0, 0] * num_aircraft),
            high=np.array([100, 100, 40000, 600, 600, 100, 360, 100, 100] * num_aircraft),
            dtype=np.float32,
        )

        # Action: Change altitude, heading, speed
        self.action_space = spaces.MultiDiscrete([3, 3, 3] * num_aircraft)

        # Matplotlib figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.ion()
        
    def seed(self, seed=None):
        return np.random.seed(100)

    def reset(self):
        """Initialize aircraft states and assign runways."""
        self.aircraft = np.zeros((self.num_aircraft, 9))

        for i in range(self.num_aircraft):
            self.aircraft[i, :7] = [
                np.random.uniform(10, 90),  # x
                np.random.uniform(10, 90),  # y
                np.random.choice(self.altitude_levels),  # altitude
                np.random.uniform(-18, 18),  # vx
                np.random.uniform(-18, 18),  # vy
                np.random.uniform(-2, 2),  # vz
                np.random.uniform(0, 360),  # heading
            ]
            self.aircraft[i, 7:] = self.runways[np.random.randint(0, len(self.runways))]

        return self.aircraft.flatten()

    def step(self, action):
        """Apply ATC actions, update positions, check for landings/collisions."""
        rewards = np.zeros(self.num_aircraft)
        done = False

        for i in range(self.num_aircraft):
            change_alt, change_heading, change_speed = action[i * 3:(i + 1) * 3]

            # Change altitude
            if change_alt == 0:
                self.aircraft[i, 2] -= 1000  # Descend
            elif change_alt == 2:
                self.aircraft[i, 2] += 1000  # Climb

            # Change heading
            if change_heading == 0:
                self.aircraft[i, 6] -= 10
            elif change_heading == 2:
                self.aircraft[i, 6] += 10

            # Change speed
            if change_speed == 0:
                self.aircraft[i, 3:5] *= 0.9
            elif change_speed == 2:
                self.aircraft[i, 3:5] *= 1.1

            # Update positions
            self.aircraft[i, 0] += self.aircraft[i, 3] * 0.1
            self.aircraft[i, 1] += self.aircraft[i, 4] * 0.1
            self.aircraft[i, 2] += self.aircraft[i, 5]

            # Check for landing
            runway_x, runway_y = self.aircraft[i, 7:]
            dist_to_runway = np.linalg.norm(self.aircraft[i, :2] - np.array([runway_x, runway_y]))

            if dist_to_runway < 1 and self.aircraft[i, 2] < 5000:  # Close enough to land
                rewards[i] += 100  # Successful landing
                done = True

        # Collision Check
        for i in range(self.num_aircraft):
            for j in range(i + 1, self.num_aircraft):
                dist_horiz = np.linalg.norm(self.aircraft[i, :2] - self.aircraft[j, :2])
                dist_vert = abs(self.aircraft[i, 2] - self.aircraft[j, 2])

                if dist_horiz < 3 and dist_vert < 1000:
                    rewards[i] -= 100
                    rewards[j] -= 100
                    done = True

        return self.aircraft.flatten(), np.sum(rewards), done, {}

    def render(self):
        """Visualize the aircraft positions, headings, and runway locations."""
        self.ax.clear()
        self.ax.set_xlim(0, self.airspace_size)
        self.ax.set_ylim(0, self.airspace_size)
        self.ax.set_title("Air Traffic Control Environment")

        # Plot runways
        for r in self.runways:
            self.ax.plot(r[0], r[1], 'ks', markersize=10, label="Runway")

        # Plot aircraft
        for i in range(self.num_aircraft):
            x, y, z, vx, vy, vz, heading, rx, ry = self.aircraft[i]

            # Aircraft marker
            self.ax.scatter(x, y, color="blue", label="Aircraft" if i == 0 else "")

            # Velocity vector (directional arrow)
            self.ax.arrow(x, y, vx * 0.05, vy * 0.05, head_width=1, head_length=1, fc='blue', ec='blue')

            # Draw path to runway
            self.ax.plot([x, rx], [y, ry], 'r--', alpha=0.5)
        
        self.ax.legend()
        plt.pause(0.1)

    def close(self):
        plt.ioff()
        plt.show()


# env = ATCEnv(num_aircraft=5)
# state = env.reset()
# rewards = []
# for i in range(100):
#     pos = [[state[i*9],state[i*9+1]] for i in range(5)]
#     print(pos)
#     action = [2,0,0] * 5
#     nextS, r, done, _ = env.step(action)
#     env.render()
#     state = nextS
#     print(f'Step {i+1} | Reward: {r}')