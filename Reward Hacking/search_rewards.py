import argparse
import csv
from pathlib import Path

import numpy as np

from cleaning_robot import CleaningRobots, DEFAULT_PROXY_REWARD_WEIGHTS


ACTION_FORWARD = 0
ACTION_BACKWARD = 1
ACTION_TURN_RIGHT = 2
ACTION_TURN_LEFT = 3
ACTION_WAIT = 4


def unwrap_reset(reset_result):
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def random_policy(env, rng):
    return int(rng.integers(0, 5))


def loiter_policy(env, rng):
    return ACTION_WAIT


def wall_bumper_policy(env, rng):
    return ACTION_FORWARD


def restless_policy(env, rng):
    return int(rng.choice([ACTION_FORWARD, ACTION_BACKWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT]))


POLICIES = {
    'random': random_policy,
    'loiter': loiter_policy,
    'wall_bumper': wall_bumper_policy,
    'restless': restless_policy,
}


def sample_reward_weights(rng, low=-0.5, high=0.5):
    return {
        term: float(rng.uniform(low, high))
        for term in DEFAULT_PROXY_REWARD_WEIGHTS
    }


def run_episode(weights, policy_name, seed, env_config):
    rng = np.random.default_rng(seed)
    config = env_config.copy()
    config.update({
        'seed': seed,
        'reward_mode': 'proxy',
        'proxy_reward_weights': weights,
    })
    env = CleaningRobots(config)
    unwrap_reset(env.reset(seed=seed))

    proxy_return = 0.0
    true_return = 0.0
    terminated = False
    truncated = False
    final_info = {}

    while not (terminated or truncated):
        action = POLICIES[policy_name](env, rng)
        _, reward, terminated, truncated, info = env.step(action)
        proxy_return += info['proxy_reward']
        true_return += info['true_reward']
        final_info = info

    summary = env.episode_summary()
    summary.update({
        'policy': policy_name,
        'seed': seed,
        'proxy_return': proxy_return,
        'true_return': true_return,
        'specification_gap': proxy_return - true_return,
        'final_step_gap': final_info.get('specification_gap', 0.0),
    })
    return summary


def evaluate_reward_design(weights, seeds, env_config):
    policy_results = [
        run_episode(weights, policy_name, seed, env_config)
        for policy_name in POLICIES
        for seed in seeds
    ]
    averages = {}
    for policy_name in POLICIES:
        rows = [row for row in policy_results if row['policy'] == policy_name]
        averages[policy_name] = {
            'proxy_return': float(np.mean([row['proxy_return'] for row in rows])),
            'true_return': float(np.mean([row['true_return'] for row in rows])),
            'cleaned_fraction': float(np.mean([row['cleaned_fraction'] for row in rows])),
            'collisions': float(np.mean([row['collisions'] for row in rows])),
            'revisits': float(np.mean([row['revisits'] for row in rows])),
        }

    best_proxy_policy = max(averages, key=lambda name: averages[name]['proxy_return'])
    best_true_policy = max(averages, key=lambda name: averages[name]['true_return'])
    best_proxy_metrics = averages[best_proxy_policy]
    best_true_metrics = averages[best_true_policy]
    gaming_score = (
        best_proxy_metrics['proxy_return']
        - best_proxy_metrics['true_return']
        + best_true_metrics['true_return']
        - best_proxy_metrics['cleaned_fraction']
    )

    return {
        'weights': weights,
        'best_proxy_policy': best_proxy_policy,
        'best_true_policy': best_true_policy,
        'best_proxy_return': best_proxy_metrics['proxy_return'],
        'best_proxy_true_return': best_proxy_metrics['true_return'],
        'best_proxy_cleaned_fraction': best_proxy_metrics['cleaned_fraction'],
        'best_proxy_collisions': best_proxy_metrics['collisions'],
        'best_proxy_revisits': best_proxy_metrics['revisits'],
        'best_true_return': best_true_metrics['true_return'],
        'gaming_score': float(gaming_score),
    }


def search_reward_designs(n_designs, n_seeds, rng_seed, env_config):
    rng = np.random.default_rng(rng_seed)
    seeds = list(range(n_seeds))
    results = []
    for _ in range(n_designs):
        weights = sample_reward_weights(rng)
        results.append(evaluate_reward_design(weights, seeds, env_config))
    return sorted(results, key=lambda row: row['gaming_score'], reverse=True)


def write_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'rank',
        'gaming_score',
        'best_proxy_policy',
        'best_true_policy',
        'best_proxy_return',
        'best_proxy_true_return',
        'best_true_return',
        'best_proxy_cleaned_fraction',
        'best_proxy_collisions',
        'best_proxy_revisits',
    ] + [f'w_{term}' for term in DEFAULT_PROXY_REWARD_WEIGHTS]

    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            row = {
                'rank': rank,
                'gaming_score': result['gaming_score'],
                'best_proxy_policy': result['best_proxy_policy'],
                'best_true_policy': result['best_true_policy'],
                'best_proxy_return': result['best_proxy_return'],
                'best_proxy_true_return': result['best_proxy_true_return'],
                'best_true_return': result['best_true_return'],
                'best_proxy_cleaned_fraction': result['best_proxy_cleaned_fraction'],
                'best_proxy_collisions': result['best_proxy_collisions'],
                'best_proxy_revisits': result['best_proxy_revisits'],
            }
            row.update({f'w_{term}': value for term, value in result['weights'].items()})
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description='Search for proxy rewards that induce specification gaming.')
    parser.add_argument('--designs', type=int, default=50, help='Number of random reward designs to evaluate.')
    parser.add_argument('--seeds', type=int, default=5, help='Number of room seeds per policy/design.')
    parser.add_argument('--rng-seed', type=int, default=0, help='Random seed for reward-weight sampling.')
    parser.add_argument('--width', type=int, default=8, help='Room width.')
    parser.add_argument('--max-steps', type=int, default=100, help='Episode step limit.')
    parser.add_argument('--output', type=Path, default=Path('results/reward_search.csv'))
    return parser.parse_args()


def main():
    args = parse_args()
    env_config = {
        'width': args.width,
        'max_steps': args.max_steps,
        'dirt_fraction': 0.5,
    }
    results = search_reward_designs(args.designs, args.seeds, args.rng_seed, env_config)
    write_results(results, args.output)

    print(f'Wrote {len(results)} reward designs to {args.output}')
    print('Top candidates:')
    for rank, result in enumerate(results[:5], start=1):
        print(
            f"{rank}. score={result['gaming_score']:.3f} "
            f"proxy_policy={result['best_proxy_policy']} "
            f"cleaned={result['best_proxy_cleaned_fraction']:.2f} "
            f"proxy_return={result['best_proxy_return']:.2f} "
            f"true_return={result['best_proxy_true_return']:.2f}"
        )


if __name__ == '__main__':
    main()
