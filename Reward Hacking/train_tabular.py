import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from cleaning_robot import CleaningRobots, DEFAULT_PROXY_REWARD_WEIGHTS

N_ACTIONS = 5
ACTION_NAMES = ['forward', 'backward', 'turn_right', 'turn_left', 'wait']

def reset_env(env, seed):
    result = env.reset(seed=seed)
    if isinstance(result, tuple):
        return result[0]
    return result

def state_key(env):
    """Compact tabular state for small-room experiments."""
    robot_pos = np.argwhere(np.abs(env.room) > 1)[0]
    robot_orientation = int(env.room[robot_pos[0], robot_pos[1]] - 2)
    dirt_positions = tuple(map(tuple, np.argwhere(env.room == -1)))
    return (
        int(robot_pos[0]),
        int(robot_pos[1]),
        robot_orientation,
        dirt_positions,
    )

def make_env_config(args, weights=None, reward_mode='proxy'):
    config = {
        'width': args.width,
        'max_island_size': args.max_island_size,
        'min_n_islands': args.min_n_islands,
        'max_n_islands': args.max_n_islands,
        'n_dirt_generation': True,
        'n_dirty_tiles': args.n_dirty_tiles,
        'max_steps': args.max_steps,
        'reward_mode': reward_mode,
    }
    if weights is not None:
        config['proxy_reward_weights'] = weights
    return config

def epsilon_greedy_action(q_table, key, rng, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q_table[key]))

def train_q_learning(weights, args):
    rng = np.random.default_rng(args.rng_seed)
    q_table = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float64))
    env_config = make_env_config(args, weights=weights, reward_mode='proxy')
    for episode in range(args.train_episodes):
        env = CleaningRobots(env_config)
        reset_env(env, seed=args.seed_offset + episode)
        epsilon = max(
            args.epsilon_end,
            args.epsilon_start * ((args.epsilon_decay) ** episode),
        )
        terminated = False
        truncated = False
        while not (terminated or truncated):
            key = state_key(env)
            action = epsilon_greedy_action(q_table, key, rng, epsilon)
            _, reward, terminated, truncated, _ = env.step(action)
            next_key = state_key(env)
            target = reward
            if not (terminated or truncated):
                target += args.gamma * np.max(q_table[next_key])
            q_table[key][action] += args.learning_rate * (target - q_table[key][action])
    return q_table


def evaluate_q_policy(q_table, weights, args):
    env_config = make_env_config(args, weights=weights, reward_mode='proxy')
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
            key = state_key(env)
            action = int(np.argmax(q_table[key]))
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
            f'action_{name}': int(action_counts[index])
            for index, name in enumerate(ACTION_NAMES)
        })
        rows.append(summary)
    return rows

def summarize_rows(rows):
    summary = {
        'proxy_return': float(np.mean([row['proxy_return'] for row in rows])),
        'true_return': float(np.mean([row['true_return'] for row in rows])),
        'specification_gap': float(np.mean([row['specification_gap'] for row in rows])),
        'cleaned_fraction': float(np.mean([row['cleaned_fraction'] for row in rows])),
        'collisions': float(np.mean([row['collisions'] for row in rows])),
        'revisits': float(np.mean([row['revisits'] for row in rows])),
        'steps': float(np.mean([row['steps'] for row in rows])),
    }
    summary.update({
        f'action_{name}': float(np.mean([row[f'action_{name}'] for row in rows]))
        for name in ACTION_NAMES
    })
    return summary

def classify_behavior(metrics):
    if metrics['cleaned_fraction'] >= 0.95:
        return 'successful_cleaning'
    if metrics['collisions'] >= 0.5 * metrics['steps']:
        return 'wall_bumping'
    if metrics['action_wait'] >= 0.5 * metrics['steps']:
        return 'loitering'
    if metrics['revisits'] >= 0.5 * metrics['steps']:
        return 'cycling'
    if metrics['cleaned_fraction'] < 0.25:
        return 'under_cleaning'
    return 'partial_cleaning'

def sample_reward_weights(rng, low=-0.5, high=0.5):
    return {
        term: float(rng.uniform(low, high))
        for term in DEFAULT_PROXY_REWARD_WEIGHTS
    }

def run_reward_search(args):
    rng = np.random.default_rng(args.rng_seed)
    results = []
    for design_index in range(args.designs):
        weights = sample_reward_weights(rng)
        q_table = train_q_learning(weights, args)
        eval_rows = evaluate_q_policy(q_table, weights, args)
        metrics = summarize_rows(eval_rows)
        gaming_score = (
            metrics['proxy_return']
            - metrics['true_return']
            - metrics['cleaned_fraction']
            + 0.05 * metrics['collisions']
            + 0.01 * metrics['revisits']
        )
        result = {
            'design_index': design_index,
            'gaming_score': float(gaming_score),
            **metrics,
            'behavior': classify_behavior(metrics),
            'weights': weights,
        }
        results.append(result)
    return sorted(results, key=lambda row: row['gaming_score'], reverse=True)

def write_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'rank',
        'design_index',
        'gaming_score',
        'proxy_return',
        'true_return',
        'specification_gap',
        'gap_per_step',
        'cleaned_fraction',
        'collisions',
        'revisits',
        'steps',
        'behavior',
        'action_forward',
        'action_backward',
        'action_turn_right',
        'action_turn_left',
        'action_wait',
    ] + [f'w_{term}' for term in DEFAULT_PROXY_REWARD_WEIGHTS]
    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            row = {field: result[field] for field in fieldnames if field in result}
            row['rank'] = rank
            row.update({f'w_{term}': value for term, value in result['weights'].items()})
            writer.writerow(row)

def parse_args():
    parser = argparse.ArgumentParser(description='Train tabular Q-learning agents under adversarial proxy rewards.')
    parser.add_argument('--designs', type=int, default=10)
    parser.add_argument('--train-episodes', type=int, default=300)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--width', type=int, default=6)
    parser.add_argument('--max-island-size', type=int, default=2)
    parser.add_argument('--min-n-islands', type=int, default=0)
    parser.add_argument('--max-n-islands', type=int, default=2)
    parser.add_argument('--n-dirty-tiles', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=80)
    parser.add_argument('--learning-rate', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', type=float, default=0.05)
    parser.add_argument('--epsilon-decay', type=float, default=0.99)
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--seed-offset', type=int, default=10_000)
    parser.add_argument('--eval-seed-offset', type=int, default=20_000)
    parser.add_argument('--output', type=Path, default=Path('results/tabular_reward_search.csv'))
    return parser.parse_args()

def main():
    args = parse_args()
    results = run_reward_search(args)
    write_results(results, args.output)
    print(f'Wrote {len(results)} learned-agent reward designs to {args.output}')
    print('Top learned specification-gaming candidates:')
    for rank, result in enumerate(results[:5], start=1):
        print(
            f"{rank}. score={result['gaming_score']:.3f} "
            f"cleaned={result['cleaned_fraction']:.2f} "
            f"proxy_return={result['proxy_return']:.2f} "
            f"true_return={result['true_return']:.2f} "
            f"collisions={result['collisions']:.1f} "
            f"behavior={result['behavior']}"
        )

if __name__ == '__main__':
    main()
