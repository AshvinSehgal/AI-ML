# Experimental Results Summary

## Main Result: Feature-Based Q Learning

The feature-based linear Q learner provides the clearest current evidence for specification gaming. Under the aligned reward, the learned policy cleans successfully. Under several adversarial proxy rewards, the same learner receives high proxy reward while failing the true cleaning objective.

| Proxy reward | Learned behavior | Learned cleaned | Expert cleaned | Proxy-true gap |
| --- | --- | --- | --- | --- |
| aligned | successful_cleaning | 1.00 | 1.00 | -0.00 |
| lazy_completion | successful_cleaning | 1.00 | 1.00 | -1.87 |
| collision_seeking | wall_bumping | 0.00 | 1.00 | 7.25 |
| dirt_avoidant | loitering | 0.00 | 1.00 | 10.29 |
| motion_seeking | cycling | 0.00 | 1.00 | 7.06 |

Key takeaways:

- The expert baseline cleans every evaluated room, so failures are not caused by unsolvable maps.
- The aligned and lazy-completion presets produce successful cleaning in this learner.
- The collision-seeking, dirt-avoidant, and motion-seeking presets produce zero cleaning with distinct failure modes.
- The strongest proxy-true gaps occur when proxy reward directly values pathological behavior such as colliding, leaving dirt, or cycling.

## Neural RL Check: DQN

DQN is noisier than the feature-based learner, but the aggregate still supports the central trend: aligned reward performs best, while adversarial proxy rewards induce low true cleaning and high proxy-true gaps.

| Proxy reward | Behaviors across seeds | Cleaned fraction | True return | Spec. gap |
| --- | --- | --- | --- | --- |
| aligned | cycling, cycling, wall_bumping | 0.58 +/- 0.07 | 1.76 +/- 0.23 | -0.96 +/- 0.42 |
| lazy_completion | loitering, loitering, loitering | 0.18 +/- 0.07 | 0.13 +/- 0.22 | 2.21 +/- 0.91 |
| motion_seeking | wall_bumping, wall_bumping, wall_bumping | 0.08 +/- 0.04 | -0.27 +/- 0.12 | 9.81 +/- 0.09 |
| collision_seeking | wall_bumping, wall_bumping, wall_bumping | 0.04 +/- 0.02 | -0.37 +/- 0.05 | 14.78 +/- 0.02 |
| dirt_avoidant | loitering, cycling, loitering | 0.00 +/- 0.00 | -0.50 +/- 0.00 | 9.63 +/- 0.67 |

Interpretation:

- DQN should be reported as a secondary robustness check rather than the main result until aligned performance is stronger.
- DQN reliably exposes reward-gaming behavior for collision-seeking, motion-seeking, and dirt-avoidant proxies.
- Multi-seed aggregation is necessary because single-seed DQN performance is unstable.

## Draft Results Paragraph

We evaluate adversarial reward designs in a cleaning-robot gridworld by training agents on proxy rewards and evaluating them against a fixed true cleaning objective. A shortest-path expert achieves perfect cleaning across all reward presets, confirming that the benchmark instances are solvable. In contrast, a feature-based Q-learning agent exhibits strong reward-dependent behavior. With the aligned reward, the agent cleans all dirt, matching expert performance. However, under adversarial proxy rewards, the same learner discovers policies that maximize proxy return while achieving zero true cleaning: collision-seeking rewards produce wall-bumping, dirt-avoidant rewards produce loitering, and motion-seeking rewards produce cycling. These results demonstrate that adversarial reward design can systematically elicit qualitatively distinct forms of specification gaming in a controlled reinforcement-learning environment. A raw-observation DQN baseline shows the same qualitative trend but with higher variance, suggesting that neural-policy results should be aggregated over seeds and treated as secondary evidence in the current implementation.

## Current Recommendation

Use the feature-based Q-learning table as the main paper result. Use DQN as a robustness check. The next technical improvement should be stronger DQN/PPO training, but the current project already has a coherent first experimental story.
