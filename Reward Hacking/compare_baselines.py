import argparse
import csv
from pathlib import Path

def read_csv_by_preset(path):
    with path.open(newline='') as handle:
        return {
            row['preset']: row
            for row in csv.DictReader(handle)
        }

def to_float(row, key):
    return float(row[key])

def compare_baselines(learned_path, expert_path):
    learned_rows = read_csv_by_preset(learned_path)
    expert_rows = read_csv_by_preset(expert_path)
    presets = sorted(set(learned_rows) & set(expert_rows))
    results = []
    for preset in presets:
        learned = learned_rows[preset]
        expert = expert_rows[preset]
        learned_cleaned = to_float(learned, 'cleaned_fraction')
        expert_cleaned = to_float(expert, 'cleaned_fraction')
        learned_true = to_float(learned, 'true_return')
        expert_true = to_float(expert, 'true_return')
        learned_proxy = to_float(learned, 'proxy_return')
        results.append({
            'preset': preset,
            'learned_behavior': learned['behavior'],
            'learned_proxy_return': learned_proxy,
            'learned_true_return': learned_true,
            'learned_cleaned_fraction': learned_cleaned,
            'expert_true_return': expert_true,
            'expert_cleaned_fraction': expert_cleaned,
            'true_return_gap_to_expert': expert_true - learned_true,
            'cleaned_fraction_gap_to_expert': expert_cleaned - learned_cleaned,
            'proxy_true_gap': learned_proxy - learned_true,
        })
    return sorted(results, key=lambda row: row['cleaned_fraction_gap_to_expert'], reverse=True)

def write_results(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'rank',
        'preset',
        'learned_behavior',
        'learned_proxy_return',
        'learned_true_return',
        'learned_cleaned_fraction',
        'expert_true_return',
        'expert_cleaned_fraction',
        'true_return_gap_to_expert',
        'cleaned_fraction_gap_to_expert',
        'proxy_true_gap',
    ]
    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            output_row = row.copy()
            output_row['rank'] = rank
            writer.writerow(output_row)

def parse_args():
    parser = argparse.ArgumentParser(description='Compare learned tabular baselines against the expert baseline.')
    parser.add_argument('--learned', type=Path, default=Path('results/preset_baselines.csv'))
    parser.add_argument('--expert', type=Path, default=Path('results/expert_baselines.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/baseline_comparison.csv'))
    return parser.parse_args()

def main():
    args = parse_args()
    rows = compare_baselines(args.learned, args.expert)
    write_results(rows, args.output)
    print(f'Wrote {len(rows)} baseline comparisons to {args.output}')
    print('Largest learned-vs-expert cleaning gaps:')
    for rank, row in enumerate(rows[:5], start=1):
        print(
            f"{rank}. {row['preset']} "
            f"behavior={row['learned_behavior']} "
            f"learned_cleaned={row['learned_cleaned_fraction']:.2f} "
            f"expert_cleaned={row['expert_cleaned_fraction']:.2f} "
            f"proxy_true_gap={row['proxy_true_gap']:.2f}"
        )

if __name__ == '__main__':
    main()
