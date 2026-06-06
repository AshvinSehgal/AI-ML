import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from cleaning_robot import PROXY_REWARD_PRESETS
from train_dqn import run_presets

METRIC_COLUMNS = [
    'proxy_return',
    'true_return',
    'specification_gap',
    'cleaned_fraction',
    'collisions',
    'revisits',
    'steps',
]

def candidate_args(args, seed_index):
    return SimpleNamespace(
        presets=args.presets,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        width=args.width,
        max_island_size=args.max_island_size,
        min_n_islands=args.min_n_islands,
        max_n_islands=args.max_n_islands,
        n_dirty_tiles=args.n_dirty_tiles,
        max_steps=args.max_steps,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        target_update=args.target_update,
        grad_clip=args.grad_clip,
        aligned_shaping=args.aligned_shaping,
        shaping_cleaned_bonus=args.shaping_cleaned_bonus,
        shaping_collision_penalty=args.shaping_collision_penalty,
        shaping_wait_penalty=args.shaping_wait_penalty,
        shaping_dirt_penalty=args.shaping_dirt_penalty,
        rng_seed=args.rng_seed + seed_index,
        seed_offset=args.seed_offset + args.seed_stride * seed_index,
        eval_seed_offset=args.eval_seed_offset,
        device=args.device,
        output=args.output,
    )


def run_seed_aggregation(args):
    per_seed_rows = []
    for seed_index in range(args.seeds):
        rows = run_presets(candidate_args(args, seed_index))
        for row in rows:
            copied = row.copy()
            copied['seed_index'] = seed_index
            copied.pop('weights', None)
            per_seed_rows.append(copied)
    aggregate_rows = []
    for preset in args.presets:
        preset_rows = [row for row in per_seed_rows if row['preset'] == preset]
        aggregate = {'preset': preset}
        for metric in METRIC_COLUMNS:
            values = np.array([row[metric] for row in preset_rows], dtype=np.float64)
            aggregate[f'{metric}_mean'] = float(np.mean(values))
            aggregate[f'{metric}_std'] = float(np.std(values))
        behaviors = [row['behavior'] for row in preset_rows]
        aggregate['behaviors'] = '|'.join(behaviors)
        aggregate_rows.append(aggregate)
    return (sorted(aggregate_rows, key=lambda row: row['cleaned_fraction_mean'], reverse=True), per_seed_rows)

def write_aggregate(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['rank', 'preset', 'behaviors']
    for metric in METRIC_COLUMNS:
        fieldnames.extend([f'{metric}_mean', f'{metric}_std'])
    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            output_row = row.copy()
            output_row['rank'] = rank
            writer.writerow(output_row)

def write_per_seed(rows, output_path):
    fieldnames = ['seed_index', 'preset', 'behavior', *METRIC_COLUMNS]
    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})

def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate DQN preset baselines across multiple training seeds.')
    parser.add_argument('--presets', nargs='+', default=sorted(PROXY_REWARD_PRESETS))
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--seed-stride', type=int, default=10_000)
    parser.add_argument('--train-episodes', type=int, default=300)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--width', type=int, default=5)
    parser.add_argument('--max-island-size', type=int, default=2)
    parser.add_argument('--min-n-islands', type=int, default=0)
    parser.add_argument('--max-n-islands', type=int, default=2)
    parser.add_argument('--n-dirty-tiles', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', type=float, default=0.05)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--batch-size', type=int, default=32)
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
    parser.add_argument('--output', type=Path, default=Path('results/dqn_seed_aggregate.csv'))
    parser.add_argument('--per-seed-output', type=Path, default=Path('results/dqn_seed_runs.csv'))
    return parser.parse_args()

def main():
    args = parse_args()
    aggregate_rows, per_seed_rows = run_seed_aggregation(args)
    write_aggregate(aggregate_rows, args.output)
    write_per_seed(per_seed_rows, args.per_seed_output)
    print(f'Wrote DQN seed aggregate to {args.output}')
    print(f'Wrote per-seed DQN rows to {args.per_seed_output}')
    print('DQN aggregate results:')
    for rank, row in enumerate(aggregate_rows, start=1):
        print(
            f"{rank}. {row['preset']} "
            f"cleaned={row['cleaned_fraction_mean']:.2f}+/-{row['cleaned_fraction_std']:.2f} "
            f"true={row['true_return_mean']:.2f}+/-{row['true_return_std']:.2f} "
            f"behaviors={row['behaviors']}"
        )

if __name__ == '__main__':
    main()
