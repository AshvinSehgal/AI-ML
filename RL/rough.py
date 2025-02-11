from PIL import Image
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO

from Envs.DroneEnvContinuous import DroneEnvContinuous

folder_path = "DroneEnv-PPO"
subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
last_subfolder = sorted(subfolders)[-1]
env = DroneEnvContinuous()
actor = tf.keras.models.load_model(os.path.join(folder_path,last_subfolder,'actor.keras'), custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})

# Parameters
output_folder = "frames"
gif_name = "drone_simulation.gif"
max_steps = 1000

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize global variables
state = env.reset()
r = 0
steps = 0

# Plot settings
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

# Main loop to generate frames
while steps < max_steps:
    mean, std = actor(np.atleast_2d(state), training=False)
    dist = tfp.distributions.Normal(mean[0], std[0])
    action = tf.clip_by_value(dist.sample(), env.action_min, env.action_max).numpy()
    # action = np.random.rand(2) * 2 - 1
    nextS, reward, done = env.step(action)
    r += reward
    steps += 1
    save_frame(steps)
    if done:
        break

# Save all frames as a GIF
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
