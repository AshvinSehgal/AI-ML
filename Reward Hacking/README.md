# Adversarial Reward Design for Specification Gaming

This project studies reward hacking in a small cleaning-robot reinforcement learning environment.
The central experiment is to separate:

- `proxy_reward`: the reward an agent is trained to optimize.
- `true_reward`: the objective we actually care about, cleaning dirt efficiently.

Specification gaming occurs when an agent achieves high proxy return while performing poorly on the true cleaning objective.

## Current Files

- `cleaning_robot.py`: cleaning-robot environment with configurable proxy rewards, true rewards, and per-step metrics.
- `search_rewards.py`: fast no-training reward search using simple behavioral policies.
- `train_tabular.py`: tabular Q-learning search over reward designs for learned-agent specification gaming.

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

Run a small learned-agent reward search:

```bash
python3 train_tabular.py --designs 10 --train-episodes 300 --eval-episodes 20 --width 6 --n-dirty-tiles 4 --max-steps 80 --output results/tabular_reward_search.csv
```

Run smoke checks:

```bash
python3 -m py_compile cleaning_robot.py search_rewards.py train_tabular.py
python3 search_rewards.py --designs 5 --seeds 2 --width 8 --max-steps 20 --output results/smoke_reward_search.csv
python3 train_tabular.py --designs 3 --train-episodes 30 --eval-episodes 5 --width 5 --n-dirty-tiles 3 --max-steps 30 --output results/smoke_tabular_reward_search.csv
```

## Paper-Shaped Next Steps

1. Establish aligned-reward baselines.
2. Search for adversarial proxy rewards.
3. Train agents under each proxy reward.
4. Evaluate every trained agent with true cleaning metrics.
5. Report proxy return, true return, cleaned fraction, collisions, revisits, and specification gap.
6. Add visual rollouts for the most interpretable failure modes.
