import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from cleaning_robot import DEFAULT_PROXY_REWARD_WEIGHTS, PROXY_REWARD_PRESETS, resolve_proxy_reward_weights
from train_tabular import classify_behavior, evaluate_q_policy, summarize_rows, train_q_learning

def build_training_args(args):
    return SimpleNamespace(
        width=args.width,
        max_island_size=args.max_island_size,
        min_n_islands=args.min_n_islands,
        max_n_islands=args.max_n_islands,
        n_dirty_tiles=args.n_dirty_tiles,
        max_steps=args.max_steps,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        rng_seed=args.rng_seed,
        seed_offset=args.seed_offset,
        eval_seed_offset=args.eval_seed_offset,
    )

def run_baselines(args):
    training_args = build_training_args(args)
    rows = []
    for preset in args.presets:
        weights = resolve_proxy_reward_weights(preset)
        q_table = train_q_learning(weights, training_args)
        eval_rows = evaluate_q_policy(q_table, weights, training_args)
        metrics = summarize_rows(eval_rows)
        gaming_score = (
            metrics['proxy_return']
            - metrics['true_return']
            - metrics['cleaned_fraction']
            + 0.05 * metrics['collisions']
            + 0.01 * metrics['revisits']
        )
        rows.append({
            'preset': preset,
            'gaming_score': float(gaming_score),
            'behavior': classify_behavior(metrics),
            **metrics,
            'weights': weights,
        })
    return sorted(rows, key=lambda row: row['gaming_score'], reverse=True)

def write_results(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'rank',
        'preset',
        'gaming_score',
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
    parser = argparse.ArgumentParser(description='Train and evaluate tabular agents on named proxy reward presets.')
    parser.add_argument('--presets', nargs='+', default=sorted(PROXY_REWARD_PRESETS))
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
    parser.add_argument('--seed-offset', type=int, default=30_000)
    parser.add_argument('--eval-seed-offset', type=int, default=40_000)
    parser.add_argument('--output', type=Path, default=Path('results/preset_baselines.csv'))
    return parser.parse_args()

def main():
    args = parse_args()
    rows = run_baselines(args)
    write_results(rows, args.output)
    print(f'Wrote {len(rows)} preset baselines to {args.output}')
    print('Preset baseline ranking:')
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank}. {row['preset']} "
            f"score={row['gaming_score']:.3f} "
            f"cleaned={row['cleaned_fraction']:.2f} "
            f"proxy_return={row['proxy_return']:.2f} "
            f"true_return={row['true_return']:.2f} "
            f"behavior={row['behavior']}"
        )

if __name__ == '__main__':
    main()
