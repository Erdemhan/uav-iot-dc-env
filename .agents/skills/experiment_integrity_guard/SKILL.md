---
name: experiment_integrity_guard
description: Validates configurations and training setups to ensure a fair comparison between Baseline (QJC), PPO, and DQN algorithms.
---

# Experiment Integrity Guard Skill

You have triggered the `experiment_integrity_guard` skill because config files, training scripts, or environments are being modified.

## Objectives
Ensure fair algorithmic comparisons (Baseline QJC vs PPO vs DQN vs PPO-LSTM) and maintain strict reproducibility.

## Guidelines & Rules

1. **Fair Comparison Requirements**:
   - **Identical Environment**: All algorithms must use the exact same observation and action spaces.
   - **Action Spaces Consistency**: Ensure `GlobalConfig.FLATTEN_ACTIONS = True` (critical for DQN compatibility) is respected by all trainers.
   - **Equal Training Budget**: All models must be trained for the same total steps/iterations defined by `GlobalConfig.TRAIN_ITERATIONS`.
   - **Unified Seed**: Centralized random seed `GlobalConfig.RANDOM_SEED` must be used across all components.

2. **Reward Function Exploitation Check**:
   - The tracking reward for the jammer MUST require active power usage.
   - Every time reward logic is modified, ensure that a condition resembling the following is enforced:
     ```python
     if (jammer_channel == uav_channel and jammer_power > 0.01):
         reward_tracking = 0.5
     ```
   - **Reason**: Prevents "zero-power channel tracking" exploits where the jammer tracks the UAV channel while broadcasting zero power.

3. **Modularity Guidelines**:
   - Keep environment logic strictly inside `simulation/pettingzoo_env.py`.
   - Keep physics/mathematical formulas (path loss, capacity, SINR) in `core/physics.py` as stateless pure functions.
   - Training scripts must remain isolated: `train_baseline.py` (QJC), `train.py` (PPO), `train_dqn.py` (DQN), and `train_ppo_lstm.py` (PPO-LSTM).
