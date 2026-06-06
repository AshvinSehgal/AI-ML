import argparse
import csv
import random
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from cleaning_robot import DEFAULT_PROXY_REWARD_WEIGHTS, PROXY_REWARD_PRESETS, CleaningRobots, resolve_proxy_reward_weights
from train_tabular import ACTION_NAMES, N_ACTIONS, classify_behavior, summarize_rows

def reset_env(env, seed):
    result = env.reset(seed=seed)
    if isinstance(result, tuple):
        return result[0]
    return result

def make_env_config(args, reward_weights):
    return {
        'width': args.width,
        'max_island_size': args.max_island_size,
        'min_n_islands': args.min_n_islands,
        'max_n_islands': args.max_n_islands,
        'n_dirt_generation': True,
        'n_dirty_tiles': args.n_dirty_tiles,
        'max_steps': args.max_steps,
        'reward_mode': 'proxy',
        'proxy_reward_weights': reward_weights,
    }

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, observation_shape, hidden_size=128):
        super().__init__()
        input_size = int(np.prod(observation_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, N_ACTIONS),
        )
        
    def forward(self, x):
        return self.net(x)

def obs_to_float(obs):
    return obs.astype(np.float32)

def choose_action(policy_net, obs, epsilon, device):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs_to_float(obs), dtype=torch.float32, device=device).unsqueeze(0)
        return int(torch.argmax(policy_net(obs_tensor), dim=1).item())

def optimize(policy_net, target_net, optimizer, replay, args, device):
    if len(replay) < args.batch_size:
        return None
    states, actions, rewards, next_states, dones = replay.sample(args.batch_size)
    states = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
    q_values = policy_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1).values
        targets = rewards + args.gamma * next_q_values * (1.0 - dones)
    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), args.grad_clip)
    optimizer.step()
    return float(loss.item())

def training_reward(proxy_reward, info, args):
    reward = proxy_reward
    if args.aligned_shaping:
        reward += args.shaping_cleaned_bonus * info['cleaned_tiles']
        reward -= args.shaping_collision_penalty * int(info['collision'])
        reward -= args.shaping_wait_penalty * int(info['reward_terms']['wait'])
        reward -= args.shaping_dirt_penalty * info['dirt_remaining']
    return reward

def train_dqn_for_preset(preset, args):
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    device = torch.device(args.device)
    reward_weights = resolve_proxy_reward_weights(preset)
    env_config = make_env_config(args, reward_weights)
    sample_env = CleaningRobots(env_config)
    sample_obs = reset_env(sample_env, seed=args.seed_offset)
    policy_net = DQN(sample_obs.shape, hidden_size=args.hidden_size).to(device)
    target_net = DQN(sample_obs.shape, hidden_size=args.hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.replay_size)
    for episode in range(args.train_episodes):
        env = CleaningRobots(env_config)
        obs = reset_env(env, seed=args.seed_offset + episode)
        epsilon = max(args.epsilon_end, args.epsilon_start * (args.epsilon_decay ** episode))
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = choose_action(policy_net, obs, epsilon, device)
            next_obs, proxy_reward, terminated, truncated, info = env.step(action)
            reward = training_reward(proxy_reward, info, args)
            done = terminated or truncated
            replay.append((obs_to_float(obs), action, float(reward), obs_to_float(next_obs), done))
            obs = next_obs
            optimize(policy_net, target_net, optimizer, replay, args, device)
        if (episode + 1) % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    return policy_net, reward_weights

def evaluate_policy(policy_net, reward_weights, args):
    device = torch.device(args.device)
    env_config = make_env_config(args, reward_weights)
    rows = []
    for index in range(args.eval_episodes):
        seed = args.eval_seed_offset + index
        env = CleaningRobots(env_config)
        obs = reset_env(env, seed=seed)
        proxy_return = 0.0
        true_return = 0.0
        action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
        ep_len = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = choose_action(policy_net, obs, epsilon=0.0, device=device)
            action_counts[action] += 1
            obs, _, terminated, truncated, info = env.step(action)
            proxy_return += info['proxy_reward']
            true_return += info['true_reward']
            ep_len += 1
        summary = env.episode_summary()
        summary.update({
            'seed': seed,
            'proxy_return': proxy_return,
            'true_return': true_return,
            'specification_gap': proxy_return - true_return,
            'gap_per_step': (proxy_return - true_return) / max(1, ep_len)
        })
        summary.update({
            f'action_{name}': int(action_counts[action_index])
            for action_index, name in enumerate(ACTION_NAMES)
        })
        rows.append(summary)
    return rows

def run_presets(args):
    rows = []
    for preset in args.presets:
        policy_net, reward_weights = train_dqn_for_preset(preset, args)
        eval_rows = evaluate_policy(policy_net, reward_weights, args)
        metrics = summarize_rows(eval_rows)
        rows.append({
            'preset': preset,
            'behavior': classify_behavior(metrics),
            **metrics,
            'weights': reward_weights,
        })
    return sorted(rows, key=lambda row: row['true_return'], reverse=True)

def write_results(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'rank',
        'preset',
        'behavior',
        'proxy_return',
        'true_return',
        'specification_gap',
        'cleaned_fraction',
        'collisions',
        'revisits',
        'steps',
        'action_forward',
        'action_backward',
        'action_turn_right',
        'action_turn_left',
        'action_wait',
    ] + [f'w_{term}' for term in DEFAULT_PROXY_REWARD_WEIGHTS]
    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(rows, start=1):
            row = {field: result[field] for field in fieldnames if field in result}
            row['rank'] = rank
            row.update({f'w_{term}': value for term, value in result['weights'].items()})
            writer.writerow(row)

def parse_args():
    parser = argparse.ArgumentParser(description='Train raw-observation DQN policies under proxy reward presets.')
    parser.add_argument('--presets', nargs='+', default=sorted(PROXY_REWARD_PRESETS))
    parser.add_argument('--train-episodes', type=int, default=500)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--width', type=int, default=5)
    parser.add_argument('--max-island-size', type=int, default=2)
    parser.add_argument('--min-n-islands', type=int, default=0)
    parser.add_argument('--max-n-islands', type=int, default=2)
    parser.add_argument('--n-dirty-tiles', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', type=float, default=0.05)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--replay-size', type=int, default=20_000)
    parser.add_argument('--target-update', type=int, default=25)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--aligned-shaping', action='store_true')
    parser.add_argument('--shaping-cleaned-bonus', type=float, default=1.0)
    parser.add_argument('--shaping-collision-penalty', type=float, default=0.2)
    parser.add_argument('--shaping-wait-penalty', type=float, default=0.02)
    parser.add_argument('--shaping-dirt-penalty', type=float, default=0.0)
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--seed-offset', type=int, default=60_000)
    parser.add_argument('--eval-seed-offset', type=int, default=40_000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('results/dqn_preset_baselines.csv'))
    return parser.parse_args()

def main():
    args = parse_args()
    rows = run_presets(args)
    write_results(rows, args.output)
    print(f'Wrote {len(rows)} DQN preset baselines to {args.output}')
    print('DQN preset results:')
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank}. {row['preset']} "
            f"cleaned={row['cleaned_fraction']:.2f} "
            f"proxy_return={row['proxy_return']:.2f} "
            f"true_return={row['true_return']:.2f} "
            f"behavior={row['behavior']}"
        )

if __name__ == '__main__':
    main()
