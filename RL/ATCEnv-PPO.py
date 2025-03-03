from PIL import Image
from Envs.ATC import ATCEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

vec_env = make_vec_env(lambda: ATCEnv(num_aircraft=5), n_envs=5)
model = PPO("MlpPolicy", vec_env, learning_rate=0.0003, n_steps=2048, batch_size=2048, verbose=1)
model.learn(total_timesteps=1000000)
model.save('ATCEnvPPO')

del model

output_folder = "frames"
gif_name = "ATCEnv-PPO.gif"
max_steps = 1000
model = PPO.load('ATCEnvPPO')
os.makedirs(output_folder, exist_ok=True)
env = ATCEnv(num_aircraft=5)
fig, ax = plt.subplots(figsize=(5, 5))
images = []

def save_frame(step):
    ax.clear()
    ax.set_xlim(0, env.airspace_size)
    ax.set_ylim(0, env.airspace_size)
    ax.set_title("Air Traffic Control Environment")

    # Plot runways
    for r in env.runways:
        env.ax.plot(r[0], r[1], 'ks', markersize=10, label="Runway")

    # Plot aircraft
    for i in range(env.num_aircraft):
        x, y, z, vx, vy, vz, heading, rx, ry = env.aircraft[i]

        # Aircraft marker
        ax.scatter(x, y, color="blue", label="Aircraft" if i == 0 else "")

        # Velocity vector (directional arrow)
        ax.arrow(x, y, vx * 0.05, vy * 0.05, head_width=1, head_length=1, fc='blue', ec='blue')

        # Draw path to runway
        ax.plot([x, rx], [y, ry], 'r--', alpha=0.5)
    
    ax.legend()
    plt.pause(0.1)
    
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
    duration=6,  # Duration of each frame in milliseconds
    loop=0  # Infinite loop
)

# Clean up the frame folder if desired
for file in os.listdir(output_folder):
    os.remove(os.path.join(output_folder, file))
os.rmdir(output_folder)

print(f"GIF saved as {gif_name}")

# obs = vec_env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs)
#     obs, reward, done, info, _ = vec_env.step(action)
#     vec_env.render(mode='human')
#     if done:
#         break
# vec_env.close()