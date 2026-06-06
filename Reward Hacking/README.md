# Adversarial Reward Design for Specification Gaming

This project studies reward hacking in a small cleaning-robot reinforcement learning environment.
The central experiment is to separate:

- `proxy_reward`: the reward an agent is trained to optimize.
- `true_reward`: the objective we actually care about, cleaning dirt efficiently.

Specification gaming occurs when an agent achieves high proxy return while performing poorly on the true cleaning objective.

## Current Files

- `cleaning_robot.py`: cleaning-robot environment with configurable proxy rewards, true rewards, and per-step metrics.
- `resource_collection.py`: resource-collection environment where useful resources are finite and useless resources can be farmed.
- `resource_experiment.py`: scripted-policy check for useful collection vs useless-resource farming.
- `search_rewards.py`: fast no-training reward search using simple behavioral policies.
- `train_tabular.py`: tabular Q-learning search over reward designs for learned-agent specification gaming.
- `train_linear.py`: feature-based linear Q-learning baseline that generalizes across room layouts better than tabular Q-learning.
- `train_dqn.py`: PyTorch DQN baseline trained directly from the environment observation channels.
- `aggregate_dqn_seeds.py`: repeats tuned DQN training across multiple seeds and reports aggregate metrics.
- `run_preset_baselines.py`: trains tabular agents on named reward presets and writes a baseline comparison table.
- `evaluate_expert.py`: evaluates a shortest-path expert policy as a solvability and upper-bound baseline.
- `compare_baselines.py`: combines learned-agent and expert results into a paper-friendly comparison table.
- `RESULTS_SUMMARY.md`: generated summary of the current experiment tables and draft results paragraph.
- `PAPER_DRAFT.md`: first narrative draft of the paper structure and main claims.

## Reward Terms

Proxy rewards are weighted combinations of these terms:

- `cleaned_tile`: number of dirty tiles cleaned this step.
- `dirt_remaining`: number of dirty tiles left after the step.
- `movement`: forward movement action.
- `backward`: backward movement action.
- `turn`: left or right turn action.
- `wait`: no-op action.
- `collision`: failed movement into a wall or obstacle.
- `revisit`: robot enters a previously visited tile.
- `done`: room is fully clean.

## Named Proxy Reward Presets

`cleaning_robot.py` includes a few hand-written proxy designs:

- `aligned`: close to the intended objective.
- `motion_seeking`: rewards movement and revisits.
- `collision_seeking`: rewards bumping into walls.
- `dirt_avoidant`: rewards leaving dirt behind and penalizes cleaning.
- `lazy_completion`: rewards waiting and cheap completion signals.

## Useful Commands

Run a fast behavior-policy reward search:

```bash
python3 search_rewards.py --designs 50 --seeds 5 --width 8 --max-steps 100 --output results/reward_search.csv
```

Run the resource-collection environment check:

```bash
python3 resource_experiment.py --seeds 10 --width 8 --n-useful-resources 4 --max-steps 60 --output results/resource_collection.csv
```

Run a small learned-agent reward search:

```bash
python3 train_tabular.py --designs 10 --train-episodes 300 --eval-episodes 20 --width 6 --n-dirty-tiles 4 --max-steps 80 --output results/tabular_reward_search.csv
```

Run the preferred lightweight learned-agent baseline:

```bash
python3 train_linear.py --train-episodes 500 --eval-episodes 10 --width 5 --n-dirty-tiles 3 --max-steps 50 --output results/linear_preset_baselines.csv
```

Run a raw-observation DQN baseline:

```bash
python3 train_dqn.py --train-episodes 500 --eval-episodes 10 --width 5 --n-dirty-tiles 3 --max-steps 50 --output results/dqn_preset_baselines.csv
```

Aggregate tuned DQN across seeds:

```bash
python3 aggregate_dqn_seeds.py --seeds 3 --train-episodes 300 --eval-episodes 10 --width 5 --n-dirty-tiles 3 --max-steps 50 --output results/dqn_seed_aggregate.csv --per-seed-output results/dqn_seed_runs.csv
```

Sweep DQN settings on the aligned reward:

```bash
python3 sweep_dqn.py --train-episodes 150 300 --learning-rates 0.001 0.0003 --hidden-sizes 64 --epsilon-decays 0.99 0.995 --shaping-options false true --eval-episodes 5 --width 5 --n-dirty-tiles 3 --max-steps 50 --batch-size 32 --output results/dqn_sweep.csv
```

Run named proxy-reward baselines:

```bash
python3 run_preset_baselines.py --train-episodes 250 --eval-episodes 10 --width 5 --n-dirty-tiles 3 --max-steps 50 --output results/preset_baselines.csv
```

Run the shortest-path expert baseline:

```bash
python3 evaluate_expert.py --eval-episodes 10 --width 5 --n-dirty-tiles 3 --max-steps 50 --output results/expert_baselines.csv
```

Compare learned baselines against the expert:

```bash
python3 compare_baselines.py --learned results/preset_baselines.csv --expert results/expert_baselines.csv --output results/baseline_comparison.csv
python3 compare_baselines.py --learned results/linear_preset_baselines.csv --expert results/expert_baselines.csv --output results/linear_baseline_comparison.csv
```

Generate a paper-style results summary:

```bash
python3 summarize_results.py --output RESULTS_SUMMARY.md
```

Run smoke checks:

```bash
python3 -m py_compile cleaning_robot.py resource_collection.py resource_experiment.py search_rewards.py train_tabular.py train_linear.py train_dqn.py sweep_dqn.py aggregate_dqn_seeds.py run_preset_baselines.py evaluate_expert.py compare_baselines.py summarize_results.py
python3 resource_experiment.py --seeds 3 --width 8 --n-useful-resources 4 --max-steps 40 --output results/smoke_resource_collection.csv
python3 search_rewards.py --designs 5 --seeds 2 --width 8 --max-steps 20 --output results/smoke_reward_search.csv
python3 train_tabular.py --designs 3 --train-episodes 30 --eval-episodes 5 --width 5 --n-dirty-tiles 3 --max-steps 30 --output results/smoke_tabular_reward_search.csv
python3 train_linear.py --train-episodes 80 --eval-episodes 5 --width 5 --n-dirty-tiles 3 --max-steps 40 --output results/smoke_linear_preset_baselines.csv
python3 train_dqn.py --presets aligned collision_seeking --train-episodes 40 --eval-episodes 3 --width 5 --n-dirty-tiles 3 --max-steps 30 --batch-size 16 --output results/smoke_dqn_preset_baselines.csv
python3 aggregate_dqn_seeds.py --presets aligned collision_seeking --seeds 2 --train-episodes 10 --eval-episodes 1 --width 5 --n-dirty-tiles 3 --max-steps 10 --batch-size 4 --output results/smoke_dqn_seed_aggregate.csv --per-seed-output results/smoke_dqn_seed_runs.csv
python3 sweep_dqn.py --train-episodes 20 --learning-rates 0.001 --hidden-sizes 64 --epsilon-decays 0.99 --shaping-options false true --eval-episodes 2 --width 5 --n-dirty-tiles 3 --max-steps 20 --batch-size 8 --output results/smoke_dqn_sweep.csv
python3 run_preset_baselines.py --train-episodes 40 --eval-episodes 5 --width 5 --n-dirty-tiles 3 --max-steps 30 --output results/smoke_preset_baselines.csv
python3 evaluate_expert.py --eval-episodes 5 --width 5 --n-dirty-tiles 3 --max-steps 30 --output results/smoke_expert_baselines.csv
python3 compare_baselines.py --learned results/preset_baselines.csv --expert results/expert_baselines.csv --output results/baseline_comparison.csv
python3 compare_baselines.py --learned results/smoke_linear_preset_baselines.csv --expert results/expert_baselines.csv --output results/smoke_linear_baseline_comparison.csv
python3 summarize_results.py --output RESULTS_SUMMARY.md
```

## Current Experimental Note

The tabular Q-learning baseline is intentionally lightweight. Early runs show a useful failure signal: bad proxy rewards can make wall-bumping highly rewarding while true cleaning stays low.

However, current tabular baselines also underperform under the `aligned` reward setting on nontrivial rooms. Treat this as a development baseline, not the final learning algorithm for the paper. The next step is to add either a curriculum/easier-room baseline or a stronger function-approximation learner.

The shortest-path expert currently reaches `cleaned_fraction = 1.0` on the same small-room setting used by the tabular baseline. This is an important control: poor learned-agent performance is not because the environment is impossible.

`results/baseline_comparison.csv` is the most useful current table for paper notes. It directly shows the gap between learned proxy-trained agents and an expert that can solve the same rooms.

`results/linear_baseline_comparison.csv` is now the strongest early result table. In the current run, the feature-based learner succeeds under `aligned` and fails under `collision_seeking`, `dirt_avoidant`, and `motion_seeking`, producing distinct failure modes: wall-bumping, loitering, and cycling.

The DQN baseline is implemented and smoke-tested. A modest aligned-only sweep reached roughly `cleaned_fraction = 0.93`, so DQN is now viable enough to test across adversarial proxy presets. The feature-based linear learner remains the clearest early result, but DQN is becoming a credible second learner.

Single-seed DQN runs are noisy: one tuned run reached high aligned performance in the sweep, while all-preset runs varied between moderate and weak aligned cleaning. Use `aggregate_dqn_seeds.py` for any DQN result you plan to cite.

Current 3-seed DQN aggregate:

- `aligned`: `cleaned_fraction = 0.58 +/- 0.07`
- `lazy_completion`: `cleaned_fraction = 0.18 +/- 0.07`
- `motion_seeking`: `cleaned_fraction = 0.08 +/- 0.04`
- `collision_seeking`: `cleaned_fraction = 0.04 +/- 0.02`
- `dirt_avoidant`: `cleaned_fraction = 0.00 +/- 0.00`

This supports the reward-gaming story, but DQN is weaker than the feature-based linear learner. For the main paper, use linear-Q as the clean primary result and DQN as a neural-policy robustness check unless further tuning improves aligned performance.

Resource collection check:

- Under `aligned`, `useful_collector` collects all useful resources and `useless_farmer` performs poorly.
- Under `useless_farming`, `useless_farmer` collects almost no useful resources but receives very high proxy reward by repeatedly farming the useless patch.
- This gives a second environment with the same specification-gaming structure as cleaning: intended goal succeeds under aligned reward, hacked behavior succeeds under adversarial proxy reward.

## Paper-Shaped Next Steps

1. Establish aligned-reward baselines.
2. Search for adversarial proxy rewards.
3. Train agents under each proxy reward.
4. Evaluate every trained agent with true cleaning metrics.
5. Report proxy return, true return, cleaned fraction, collisions, revisits, and specification gap.
6. Add visual rollouts for the most interpretable failure modes.
