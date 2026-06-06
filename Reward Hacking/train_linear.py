import argparse
import csv
from pathlib import Path
import numpy as np
from cleaning_robot import CleaningRobots, DEFAULT_PROXY_REWARD_WEIGHTS, PROXY_REWARD_PRESETS, get_robot_pos, resolve_proxy_reward_weights
from evaluate_expert import DIRECTION_VECTORS, expert_action, reset_env, shortest_path_to_dirt
from train_tabular import ACTION_NAMES, N_ACTIONS, classify_behavior, summarize_rows

def make_env_config(args, weights, reward_mode='proxy'):
    return {
        'width': args.width,
        'max_island_size': args.max_island_size,
        'min_n_islands': args.min_n_islands,
        'max_n_islands': args.max_n_islands,
        'n_dirt_generation': True,
        'n_dirty_tiles': args.n_dirty_tiles,
        'max_steps': args.max_steps,
        'reward_mode': reward_mode,
        'proxy_reward_weights': weights,
    }

def nearest_dirt_distance(room, start):
    path = shortest_path_to_dirt(room)
    if len(path) > 1 and tuple(path[0]) == tuple(start):
        return len(path) - 1
    dirt_positions = np.argwhere(room == -1)
    if len(dirt_positions) == 0:
        return 0
    return int(np.min(np.abs(dirt_positions - np.array(start)).sum(axis=1)))

def hypothetical_position(env, action):
    robot_pos = get_robot_pos(env.room)
    orientation = int(env.room[robot_pos[0], robot_pos[1]] - 2)
    if action == 0:
        move = DIRECTION_VECTORS[orientation]
    elif action == 1:
        move = -DIRECTION_VECTORS[orientation]
    else:
        move = np.array([0, 0])
    return robot_pos + move

def action_features(env, action):
    robot_pos = get_robot_pos(env.room)
    orientation = int(env.room[robot_pos[0], robot_pos[1]] - 2)
    next_pos = hypothetical_position(env, action)
    max_distance = max(1, env.width * 2)
    collision = int(action in [0, 1] and env.room[next_pos[0], next_pos[1]] == 0)
    target_dirty = int(action in [0, 1] and env.room[next_pos[0], next_pos[1]] == -1)
    revisit = int(action in [0, 1] and not collision and env.visited[next_pos[0], next_pos[1]] > 0)
    current_distance = nearest_dirt_distance(env.room, robot_pos)
    next_distance = current_distance if collision else nearest_dirt_distance(env.room, next_pos)
    distance_delta = (current_distance - next_distance) / max_distance
    expert_match = int(action == expert_action(env))
    one_hot = np.zeros(N_ACTIONS, dtype=np.float64)
    one_hot[action] = 1.0
    return np.concatenate([
        np.array([
            1.0,
            current_distance / max_distance,
            distance_delta,
            target_dirty,
            collision,
            revisit,
            expert_match,
            int(action == 4),
            orientation / 3.0,
            env.count_dirt() / max(1, env.initial_dirt_count),
        ], dtype=np.float64),
        one_hot,
    ])

def q_values(weights, env):
    return np.array([
        float(np.dot(weights, action_features(env, action)))
        for action in range(N_ACTIONS)
    ])

def choose_action(weights, env, rng, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q_values(weights, env)))

def train_linear_q(weights_config, args):
    rng = np.random.default_rng(args.rng_seed)
    env_config = make_env_config(args, weights_config)
    sample_env = CleaningRobots(env_config)
    reset_env(sample_env, seed=args.seed_offset)
    feature_count = len(action_features(sample_env, 0))
    weights = np.zeros(feature_count, dtype=np.float64)
    for episode in range(args.train_episodes):
        env = CleaningRobots(env_config)
        reset_env(env, seed=args.seed_offset + episode)
        epsilon = max(args.epsilon_end, args.epsilon_start * (args.epsilon_decay ** episode))
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = choose_action(weights, env, rng, epsilon)
            features = action_features(env, action)
            prediction = float(np.dot(weights, features))
            _, reward, terminated, truncated, _ = env.step(action)
            target = reward
            if not (terminated or truncated):
                target += args.gamma * float(np.max(q_values(weights, env)))
            td_error = np.clip(target - prediction, -args.td_clip, args.td_clip)
            weights += args.learning_rate * td_error * features
    return weights

def evaluate_linear_policy(weights, weights_config, args):
    env_config = make_env_config(args, weights_config)
    rows = []
    for index in range(args.eval_episodes):
        seed = args.eval_seed_offset + index
        env = CleaningRobots(env_config)
        reset_env(env, seed=seed)
        proxy_return = 0.0
        true_return = 0.0
        action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = int(np.argmax(q_values(weights, env)))
            action_counts[action] += 1
            _, _, terminated, truncated, info = env.step(action)
            proxy_return += info['proxy_reward']
            true_return += info['true_reward']
        summary = env.episode_summary()
        summary.update({
            'seed': seed,
            'proxy_return': proxy_return,
            'true_return': true_return,
            'specification_gap': proxy_return - true_return,
        })
        summary.update({
            f'action_{name}': int(action_counts[action_index])
            for action_index, name in enumerate(ACTION_NAMES)
        })
        rows.append(summary)
    return rows

def run_preset_baselines(args):
    rows = []
    for preset in args.presets:
        weights_config = resolve_proxy_reward_weights(preset)
        learned_weights = train_linear_q(weights_config, args)
        eval_rows = evaluate_linear_policy(learned_weights, weights_config, args)
        metrics = summarize_rows(eval_rows)
        rows.append({
            'preset': preset,
            'behavior': classify_behavior(metrics),
            **metrics,
            'weights': weights_config,
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
    parser = argparse.ArgumentParser(description='Train feature-based linear Q agents on named proxy reward presets.')
    parser.add_argument('--presets', nargs='+', default=sorted(PROXY_REWARD_PRESETS))
    parser.add_argument('--train-episodes', type=int, default=500)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--width', type=int, default=5)
    parser.add_argument('--max-island-size', type=int, default=2)
    parser.add_argument('--min-n-islands', type=int, default=0)
    parser.add_argument('--max-n-islands', type=int, default=2)
    parser.add_argument('--n-dirty-tiles', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon-start', type=float, default=0.8)
    parser.add_argument('--epsilon-end', type=float, default=0.02)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--td-clip', type=float, default=5.0)
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--seed-offset', type=int, default=50_000)
    parser.add_argument('--eval-seed-offset', type=int, default=40_000)
    parser.add_argument('--output', type=Path, default=Path('results/linear_preset_baselines.csv'))
    return parser.parse_args()

def main():
    args = parse_args()
    rows = run_preset_baselines(args)
    write_results(rows, args.output)
    print(f'Wrote {len(rows)} linear-Q preset baselines to {args.output}')
    print('Linear-Q preset results:')
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
