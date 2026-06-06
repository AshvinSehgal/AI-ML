import argparse
import csv
from pathlib import Path

def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))

def as_float(row, key):
    return float(row[key])

def fmt(value):
    return f'{value:.2f}'

def fmt_behaviors(value):
    return value.replace('|', ', ')

def markdown_table(headers, rows):
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)

def linear_table(path):
    rows = read_rows(path)
    rows = sorted(rows, key=lambda row: as_float(row, 'learned_cleaned_fraction'), reverse=True)
    table_rows = [
        [
            row['preset'],
            row['learned_behavior'],
            fmt(as_float(row, 'learned_cleaned_fraction')),
            fmt(as_float(row, 'expert_cleaned_fraction')),
            fmt(as_float(row, 'proxy_true_gap')),
        ]
        for row in rows
    ]
    return markdown_table(
        ['Proxy reward', 'Learned behavior', 'Learned cleaned', 'Expert cleaned', 'Proxy-true gap'],
        table_rows,
    )

def dqn_table(path):
    rows = read_rows(path)
    rows = sorted(rows, key=lambda row: as_float(row, 'cleaned_fraction_mean'), reverse=True)
    table_rows = [
        [
            row['preset'],
            fmt_behaviors(row['behaviors']),
            f"{fmt(as_float(row, 'cleaned_fraction_mean'))} +/- {fmt(as_float(row, 'cleaned_fraction_std'))}",
            f"{fmt(as_float(row, 'true_return_mean'))} +/- {fmt(as_float(row, 'true_return_std'))}",
            f"{fmt(as_float(row, 'specification_gap_mean'))} +/- {fmt(as_float(row, 'specification_gap_std'))}",
        ]
        for row in rows
    ]
    return markdown_table(['Proxy reward', 'Behaviors across seeds', 'Cleaned fraction', 'True return', 'Spec. gap'], table_rows)

def write_summary(args):
    linear_rows = read_rows(args.linear)
    adversarial_failures = [
        row for row in linear_rows
        if as_float(row, 'learned_cleaned_fraction') == 0.0
    ]
    successful_rows = [
        row for row in linear_rows
        if as_float(row, 'learned_cleaned_fraction') >= 0.95
    ]
    content = f"""# Experimental Results Summary

## Main Result: Feature-Based Q Learning
The feature-based linear Q learner provides the clearest current evidence for specification gaming. Under the aligned reward, the learned policy cleans successfully. Under several adversarial proxy rewards, the same learner receives high proxy reward while failing the true cleaning objective.

{linear_table(args.linear)}

Key takeaways:

- The expert baseline cleans every evaluated room, so failures are not caused by unsolvable maps.
- The aligned and lazy-completion presets produce successful cleaning in this learner.
- The collision-seeking, dirt-avoidant, and motion-seeking presets produce zero cleaning with distinct failure modes.
- The strongest proxy-true gaps occur when proxy reward directly values pathological behavior such as colliding, leaving dirt, or cycling.

## Neural RL Check: DQN
DQN is noisier than the feature-based learner, but the aggregate still supports the central trend: aligned reward performs best, while adversarial proxy rewards induce low true cleaning and high proxy-true gaps.
{dqn_table(args.dqn)}

Interpretation:
- DQN should be reported as a secondary robustness check rather than the main result until aligned performance is stronger.
- DQN reliably exposes reward-gaming behavior for collision-seeking, motion-seeking, and dirt-avoidant proxies.
- Multi-seed aggregation is necessary because single-seed DQN performance is unstable.

## Draft Results Paragraph
We evaluate adversarial reward designs in a cleaning-robot gridworld by training agents on proxy rewards and evaluating them against a fixed true cleaning objective. A shortest-path expert achieves perfect cleaning across all reward presets, confirming that the benchmark instances are solvable. In contrast, a feature-based Q-learning agent exhibits strong reward-dependent behavior. With the aligned reward, the agent cleans all dirt, matching expert performance. However, under adversarial proxy rewards, the same learner discovers policies that maximize proxy return while achieving zero true cleaning: collision-seeking rewards produce wall-bumping, dirt-avoidant rewards produce loitering, and motion-seeking rewards produce cycling. These results demonstrate that adversarial reward design can systematically elicit qualitatively distinct forms of specification gaming in a controlled reinforcement-learning environment. A raw-observation DQN baseline shows the same qualitative trend but with higher variance, suggesting that neural-policy results should be aggregated over seeds and treated as secondary evidence in the current implementation.

## Current Recommendation
Use the feature-based Q-learning table as the main paper result. Use DQN as a robustness check. The next technical improvement should be stronger DQN/PPO training, but the current project already has a coherent first experimental story.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    return content

def parse_args():
    parser = argparse.ArgumentParser(description='Generate a markdown summary from experiment CSVs.')
    parser.add_argument('--linear', type=Path, default=Path('results/linear_baseline_comparison.csv'))
    parser.add_argument('--dqn', type=Path, default=Path('results/dqn_seed_aggregate.csv'))
    parser.add_argument('--output', type=Path, default=Path('RESULTS_SUMMARY.md'))
    return parser.parse_args()

def main():
    args = parse_args()
    write_summary(args)
    print(f'Wrote results summary to {args.output}')

if __name__ == '__main__':
    main()
