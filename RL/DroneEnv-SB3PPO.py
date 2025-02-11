from PIL import Image
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Envs.DroneEnvBox import DroneEnvBox
import matplotlib.pyplot as plt
import matplotlib.patches as patches



env = DroneEnvBox()
vec_env = make_vec_env(DroneEnvBox, n_envs=5)
model = PPO("MlpPolicy", vec_env, learning_rate=0.0003, n_steps=2048, batch_size=2048, verbose=1)
model.learn(total_timesteps=10000000)
model.save('DroneEnvSB3PPO')

del model

env = DroneEnvBox()

# Parameters
output_folder = "frames"
gif_name = "DroneEnv-SB3PPO.gif"
max_steps = 1000
model = PPO.load('DroneEnvSB3PPO')
os.makedirs(output_folder, exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 5))
images = []

def save_frame(step):
    ax.clear()
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)

    ax.plot(env.dronePos[0], env.dronePos[1], 'bo', markersize=10, label="Drone")
    ax.plot(env.SAMPos[:, 0], env.SAMPos[:, 1], 'rx', markersize=8, label="SAM System")
    
    for pos in env.SAMPos:
        range_circle = patches.Circle((pos[0], pos[1]), env.range, color='r', alpha=0.2)
        ax.add_patch(range_circle)
    
    ax.plot(env.goalPos[0], env.goalPos[1], 'go', markersize=10, label="Goal")
    ax.legend(loc="upper right")
    ax.grid(False)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Drone Environment | Step: {step} | Reward: {r:.2f}")
    
    # Save the current frame
    frame_path = os.path.join(output_folder, f"frame_{step:04d}.png")
    plt.savefig(frame_path)
    images.append(Image.open(frame_path))

state = env.reset()
r = 0
steps = 0

while steps < max_steps:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    r += reward
    steps += 1
    save_frame(steps)
    if done:
        break

images[0].save(
    gif_name,
    save_all=True,
    append_images=images[1:],
    optimize=True,
    duration=100,  # Duration of each frame in milliseconds
    loop=0  # Infinite loop
)

# Clean up the frame folder if desired
for file in os.listdir(output_folder):
    os.remove(os.path.join(output_folder, file))
os.rmdir(output_folder)

print(f"GIF saved as {gif_name}")