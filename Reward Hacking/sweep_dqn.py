import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from train_dqn import run_presets

def base_args(args):
    return {
        'presets': ['aligned'],
        'eval_episodes': args.eval_episodes,
        'width': args.width,
        'max_island_size': args.max_island_size,
        'min_n_islands': args.min_n_islands,
        'max_n_islands': args.max_n_islands,
        'n_dirty_tiles': args.n_dirty_tiles,
        'max_steps': args.max_steps,
        'batch_size': args.batch_size,
        'replay_size': args.replay_size,
        'target_update': args.target_update,
        'grad_clip': args.grad_clip,
        'rng_seed': args.rng_seed,
        'seed_offset': args.seed_offset,
        'eval_seed_offset': args.eval_seed_offset,
        'device': args.device,
        'output': args.output,
    }

def candidate_configs(args):
    candidates = []
    for train_episodes in args.train_episodes:
        for learning_rate in args.learning_rates:
            for hidden_size in args.hidden_sizes:
                for epsilon_decay in args.epsilon_decays:
                    for aligned_shaping in args.shaping_options:
                        config = base_args(args)
                        config.update({
                            'train_episodes': train_episodes,
                            'hidden_size': hidden_size,
                            'learning_rate': learning_rate,
                            'gamma': args.gamma,
                            'epsilon_start': args.epsilon_start,
                            'epsilon_end': args.epsilon_end,
                            'epsilon_decay': epsilon_decay,
                            'aligned_shaping': aligned_shaping,
                            'shaping_cleaned_bonus': args.shaping_cleaned_bonus,
                            'shaping_collision_penalty': args.shaping_collision_penalty,
                            'shaping_wait_penalty': args.shaping_wait_penalty,
                            'shaping_dirt_penalty': args.shaping_dirt_penalty,
                        })
                        candidates.append(SimpleNamespace(**config))
    return candidates

def run_sweep(args):
    rows = []
    for index, candidate in enumerate(candidate_configs(args)):
        results = run_presets(candidate)
        result = results[0]
        rows.append({
            'candidate': index,
            'train_episodes': candidate.train_episodes,
            'learning_rate': candidate.learning_rate,
            'hidden_size': candidate.hidden_size,
            'epsilon_decay': candidate.epsilon_decay,
            'aligned_shaping': candidate.aligned_shaping,
            'behavior': result['behavior'],
            'proxy_return': result['proxy_return'],
            'true_return': result['true_return'],
            'cleaned_fraction': result['cleaned_fraction'],
            'collisions': result['collisions'],
            'revisits': result['revisits'],
            'steps': result['steps'],
        })
    return sorted(rows, key=lambda row: (row['cleaned_fraction'], row['true_return']), reverse=True)

def write_results(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'rank',
        'candidate',
        'train_episodes',
        'learning_rate',
        'hidden_size',
        'epsilon_decay',
        'aligned_shaping',
        'behavior',
        'proxy_return',
        'true_return',
        'cleaned_fraction',
        'collisions',
        'revisits',
        'steps',
    ]
    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            output_row = row.copy()
            output_row['rank'] = rank
            writer.writerow(output_row)

def parse_bool(value):
    value = value.lower()
    if value in ['true', '1', 'yes', 'y']:
        return True
    if value in ['false', '0', 'no', 'n']:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}.")

def parse_args():
    parser = argparse.ArgumentParser(description='Sweep DQN hyperparameters on the aligned reward preset.')
    parser.add_argument('--train-episodes', nargs='+', type=int, default=[200, 500])
    parser.add_argument('--learning-rates', nargs='+', type=float, default=[1e-3, 3e-4])
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[64, 128])
    parser.add_argument('--epsilon-decays', nargs='+', type=float, default=[0.99, 0.995])
    parser.add_argument('--shaping-options', nargs='+', type=parse_bool, default=[False, True])
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--width', type=int, default=5)
    parser.add_argument('--max-island-size', type=int, default=2)
    parser.add_argument('--min-n-islands', type=int, default=0)
    parser.add_argument('--max-n-islands', type=int, default=2)
    parser.add_argument('--n-dirty-tiles', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--replay-size', type=int, default=20_000)
    parser.add_argument('--target-update', type=int, default=25)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', type=float, default=0.05)
    parser.add_argument('--shaping-cleaned-bonus', type=float, default=1.0)
    parser.add_argument('--shaping-collision-penalty', type=float, default=0.2)
    parser.add_argument('--shaping-wait-penalty', type=float, default=0.02)
    parser.add_argument('--shaping-dirt-penalty', type=float, default=0.0)
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--seed-offset', type=int, default=70_000)
    parser.add_argument('--eval-seed-offset', type=int, default=40_000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('results/dqn_sweep.csv'))
    return parser.parse_args()

def main():
    args = parse_args()
    rows = run_sweep(args)
    write_results(rows, args.output)
    print(f'Wrote {len(rows)} DQN sweep candidates to {args.output}')
    print('Top aligned DQN candidates:')
    for rank, row in enumerate(rows[:5], start=1):
        print(
            f"{rank}. cleaned={row['cleaned_fraction']:.2f} "
            f"true_return={row['true_return']:.2f} "
            f"behavior={row['behavior']} "
            f"episodes={row['train_episodes']} "
            f"lr={row['learning_rate']} "
            f"hidden={row['hidden_size']} "
            f"decay={row['epsilon_decay']} "
            f"shaping={row['aligned_shaping']}"
        )

if __name__ == '__main__':
    main()
