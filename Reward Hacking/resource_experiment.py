import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np

from resource_collection import ResourceCollection, PROXY_REWARD_PRESETS, get_collector_pos, resolve_proxy_reward_weights


DIRECTION_VECTORS = {
    0: np.array([1, 0]),
    1: np.array([0, 1]),
    2: np.array([-1, 0]),
    3: np.array([0, -1]),
}

POLICIES = ['useful_collector', 'useless_farmer']


def reset_env(env, seed):
    result = env.reset(seed=seed)
    if isinstance(result, tuple):
        return result[0]
    return result

def shortest_path_to_value(resource_map, target_value):
    start = tuple(get_collector_pos(resource_map))
    targets = {tuple(pos) for pos in np.argwhere(resource_map == target_value)}
    if not targets:
        return [start]
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        if current in targets:
            return path
        for move in DIRECTION_VECTORS.values():
            neighbor = (current[0] + int(move[0]), current[1] + int(move[1]))
            if neighbor in visited:
                continue
            if resource_map[neighbor[0], neighbor[1]] == 0:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return [start]

def action_toward_path(env, path):
    if len(path) < 2:
        return 4
    collector_pos = np.array(path[0])
    next_pos = np.array(path[1])
    desired_move = next_pos - collector_pos
    orientation = int(env.resource_map[collector_pos[0], collector_pos[1]] - 2)
    if np.array_equal(desired_move, DIRECTION_VECTORS[orientation]):
        return 0
    if np.array_equal(desired_move, -DIRECTION_VECTORS[orientation]):
        return 1
    if np.array_equal(desired_move, DIRECTION_VECTORS[(orientation + 1) % 4]):
        return 2
    return 3

def useful_collector_policy(env):
    path = shortest_path_to_value(env.resource_map, -1)
    return action_toward_path(env, path)

def useless_farmer_policy(env):
    if env.under_collector == -2:
        return 4
    path = shortest_path_to_value(env.resource_map, -2)
    return action_toward_path(env, path)

def policy_action(policy_name, env):
    if policy_name == 'useful_collector':
        return useful_collector_policy(env)
    if policy_name == 'useless_farmer':
        return useless_farmer_policy(env)
    raise ValueError(f"Unknown policy {policy_name}.")

def make_env_config(args, preset):
    return {
        'width': args.width,
        'n_useful_resources': args.n_useful_resources,
        'max_steps': args.max_steps,
        'reward_mode': 'proxy',
        'proxy_reward_weights': resolve_proxy_reward_weights(preset),
    }

def run_episode(args, preset, policy_name, seed):
    env = ResourceCollection(make_env_config(args, preset))
    reset_env(env, seed=seed)
    proxy_return = 0.0
    true_return = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy_action(policy_name, env)
        _, _, terminated, truncated, info = env.step(action)
        proxy_return += info['proxy_reward']
        true_return += info['true_reward']
    summary = env.episode_summary()
    summary.update({
        'preset': preset,
        'policy': policy_name,
        'seed': seed,
        'proxy_return': proxy_return,
        'true_return': true_return,
        'specification_gap': proxy_return - true_return,
    })
    return summary

def summarize(rows):
    output_rows = []
    for preset in sorted(set(row['preset'] for row in rows)):
        for policy_name in POLICIES:
            policy_rows = [row for row in rows if row['preset'] == preset and row['policy'] == policy_name]
            output_rows.append({
                'preset': preset,
                'policy': policy_name,
                'proxy_return': float(np.mean([row['proxy_return'] for row in policy_rows])),
                'true_return': float(np.mean([row['true_return'] for row in policy_rows])),
                'specification_gap': float(np.mean([row['specification_gap'] for row in policy_rows])),
                'useful_fraction': float(np.mean([row['useful_fraction'] for row in policy_rows])),
                'useless_collected': float(np.mean([row['total_useless_collected'] for row in policy_rows])),
                'steps': float(np.mean([row['steps'] for row in policy_rows])),
            })
    return sorted(output_rows, key=lambda row: (row['preset'], -row['proxy_return']))

def run_experiment(args):
    rows = []
    for preset in args.presets:
        for policy_name in POLICIES:
            for seed in range(args.seeds):
                rows.append(run_episode(args, preset, policy_name, args.seed_offset + seed))
    return summarize(rows)

def write_results(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['preset', 'policy', 'proxy_return', 'true_return', 'specification_gap', 'useful_fraction', 'useless_collected', 'steps']
    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate useful collection vs useless-resource farming.')
    parser.add_argument('--presets', nargs='+', default=sorted(PROXY_REWARD_PRESETS))
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--n-useful-resources', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=60)
    parser.add_argument('--seed-offset', type=int, default=80_000)
    parser.add_argument('--output', type=Path, default=Path('results/resource_collection.csv'))
    return parser.parse_args()

def main():
    args = parse_args()
    rows = run_experiment(args)
    write_results(rows, args.output)
    print(f'Wrote resource collection results to {args.output}')
    print('Resource collection results:')
    for row in rows:
        print(
            f"{row['preset']} / {row['policy']}: "
            f"useful={row['useful_fraction']:.2f} "
            f"useless={row['useless_collected']:.1f} "
            f"proxy={row['proxy_return']:.2f} "
            f"true={row['true_return']:.2f}"
        )

if __name__ == '__main__':
    main()
