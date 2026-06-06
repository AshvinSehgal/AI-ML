import argparse
import csv
from collections import deque
from pathlib import Path
import numpy as np
from cleaning_robot import CleaningRobots, DEFAULT_PROXY_REWARD_WEIGHTS, PROXY_REWARD_PRESETS, get_robot_pos, resolve_proxy_reward_weights
from train_tabular import ACTION_NAMES, classify_behavior, summarize_rows

DIRECTION_VECTORS = {
    0: np.array([1, 0]),   # Up
    1: np.array([0, 1]),   # Right
    2: np.array([-1, 0]),  # Down
    3: np.array([0, -1]),  # Left
}

def reset_env(env, seed):
    result = env.reset(seed=seed)
    if isinstance(result, tuple):
        return result[0]
    return result

def shortest_path_to_dirt(room):
    start = tuple(get_robot_pos(room))
    dirt_targets = {tuple(pos) for pos in np.argwhere(room == -1)}
    if not dirt_targets:
        return [start]
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        if current in dirt_targets:
            return path
        for move in DIRECTION_VECTORS.values():
            neighbor = (current[0] + int(move[0]), current[1] + int(move[1]))
            if neighbor in visited:
                continue
            if room[neighbor[0], neighbor[1]] == 0:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return [start]

def expert_action(env):
    path = shortest_path_to_dirt(env.room)
    if len(path) < 2:
        return 4
    robot_pos = np.array(path[0])
    next_pos = np.array(path[1])
    desired_move = next_pos - robot_pos
    orientation = int(env.room[robot_pos[0], robot_pos[1]] - 2)
    if np.array_equal(desired_move, DIRECTION_VECTORS[orientation]):
        return 0
    if np.array_equal(desired_move, -DIRECTION_VECTORS[orientation]):
        return 1
    if np.array_equal(desired_move, DIRECTION_VECTORS[(orientation + 1) % 4]):
        return 2
    return 3

def make_env_config(args, weights):
    return {
        'width': args.width,
        'max_island_size': args.max_island_size,
        'min_n_islands': args.min_n_islands,
        'max_n_islands': args.max_n_islands,
        'n_dirt_generation': True,
        'n_dirty_tiles': args.n_dirty_tiles,
        'max_steps': args.max_steps,
        'reward_mode': 'proxy',
        'proxy_reward_weights': weights,
    }

def evaluate_expert_for_preset(preset, args):
    weights = resolve_proxy_reward_weights(preset)
    env_config = make_env_config(args, weights)
    rows = []
    for index in range(args.eval_episodes):
        seed = args.eval_seed_offset + index
        env = CleaningRobots(env_config)
        reset_env(env, seed=seed)
        proxy_return = 0.0
        true_return = 0.0
        action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = expert_action(env)
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
    metrics = summarize_rows(rows)
    return {
        'preset': preset,
        'behavior': classify_behavior(metrics),
        **metrics,
        'weights': weights,
    }

def run_expert_baselines(args):
    results = [evaluate_expert_for_preset(preset, args) for preset in args.presets]
    return sorted(results, key=lambda row: row['true_return'], reverse=True)

def write_results(results, output_path):
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
        for rank, result in enumerate(results, start=1):
            row = {field: result[field] for field in fieldnames if field in result}
            row['rank'] = rank
            row.update({f'w_{term}': value for term, value in result['weights'].items()})
            writer.writerow(row)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a shortest-path expert cleaning policy.')
    parser.add_argument('--presets', nargs='+', default=sorted(PROXY_REWARD_PRESETS))
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--width', type=int, default=6)
    parser.add_argument('--max-island-size', type=int, default=2)
    parser.add_argument('--min-n-islands', type=int, default=0)
    parser.add_argument('--max-n-islands', type=int, default=2)
    parser.add_argument('--n-dirty-tiles', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=80)
    parser.add_argument('--eval-seed-offset', type=int, default=40_000)
    parser.add_argument('--output', type=Path, default=Path('results/expert_baselines.csv'))
    return parser.parse_args()

def main():
    args = parse_args()
    results = run_expert_baselines(args)
    write_results(results, args.output)
    print(f'Wrote {len(results)} expert baselines to {args.output}')
    print('Expert baseline results:')
    for rank, row in enumerate(results, start=1):
        print(
            f"{rank}. {row['preset']} "
            f"cleaned={row['cleaned_fraction']:.2f} "
            f"proxy_return={row['proxy_return']:.2f} "
            f"true_return={row['true_return']:.2f} "
            f"behavior={row['behavior']}"
        )

if __name__ == '__main__':
    main()
