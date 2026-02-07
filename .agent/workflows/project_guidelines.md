---
description: Project guidelines for IoT-UAV intelligent jammer research
---

# Research Project Context

## Primary Objective
This project aims to publish a **conference paper** on intelligent jamming attacks in IoT-UAV data collection scenarios:

- **Baseline Reference:** Liao et al. (2025) - Q-Learning based jammer channel selection
- **Research Domain:** Security-sensitive UAV-IoT data collection networks
- **Core Contribution:** Fair algorithmic comparison of intelligent jamming strategies (Baseline QJC vs PPO vs DQN)
- **Key Innovation:** Fixed reward mechanism preventing degenerate policies + comprehensive evaluation framework

## Research Artifacts
- `liaovd(2025).pdf`: Reference paper - Baseline algorithm implementation
- `experiments/`: Fair comparison results (Baseline, PPO, DQN)
- `RAPOR.md`: Technical documentation (Turkish - thesis requirement)
- `README.md`: System workflow and architecture (English)

---

# Development Guidelines

## Language Policy
- **Code & Comments:** MUST be in **English**
- **README.md & todo.md:** MUST be in **English**
- **RAPOR.md:** MUST remain in **Turkish** (Academic thesis requirement)

## Documentation Protocol

### After Every Code Change
1. **Update RAPOR.md Section 6 (Gelişim Günlüğü)**
   - Add new timestamped version entry
   - Format: `### [DD.MM.YYYY HH:MM] - Descriptive Title (vX.Y.Z)`
   - Ordering: Chronological (oldest to newest)
   - Include: What changed, why, and technical impact

2. **Update README.md if workflow changes**
   - Maintain existing structure
   - Update: Initialization, Simulation Loop, Analysis steps
   - Add new scripts/commands to relevant sections

3. **Technology Stack Changes**
   - New library → Document in RAPOR.md Section 2.5
   - Include: What it is, why we use it, how it integrates

## Code Quality Standards

### Modularity
- Keep environment logic in `simulation/pettingzoo_env.py`
- Physics calculations in `core/physics.py` (stateless functions)
- Training scripts separate: `train.py` (PPO), `train_dqn.py` (DQN), `train_baseline.py` (QJC)

### Configuration Management
- All hyperparameters in `confs/model_config.py`
- Single source of truth: `GlobalConfig`, `PPOConfig`, `DQNConfig`, `QJCConfig`
- Random seed centralized: `GlobalConfig.RANDOM_SEED`

### Fair Comparison Requirements
- **Same environment:** All algorithms use identical observation/action spaces
- **Same budget:** Equal training iterations (`GlobalConfig.TRAIN_ITERATIONS`)
- **Same reward:** Unified reward mechanism across PPO/DQN
- **Documented differences:** API Stack (PPO Old vs DQN Old) must be justified in RAPOR.md

## Research Reproducibility

### Critical Parameters to Document
- Random seed value
- Training iterations
- Reward function components and weights
- Action space configuration (flatten_actions)
- Gamma (discount factor)

### Experiment Pipeline
- Use `run_experiments.py` for full pipeline (train + eval + compare)
- Results saved to `experiments/{baseline,ppo,dqn}/`
- Comparison plot auto-generated via `visualization/compare.py`

## Version Control Best Practices
- Commit after each logical feature completion
- Meaningful commit messages: `feat:`, `fix:`, `docs:`, `refactor:`
- Keep experiment results in `.gitignore` (large files)
- Include trained models only if necessary for paper replication

## Paper Writing Checklist
When preparing conference paper:
1. Latest `comparison_result.png` from experiments
2. Performance metrics from `history.csv` files
3. RAPOR.md technical sections as reference
4. Reward mechanism explanation (README.md Section 3)
5. Fair comparison justification (FLATTEN_ACTIONS, API Stack choices)

---

# Common Tasks

## Running Full Experiment
```bash
python run_experiments.py
# Trains Baseline, PPO, DQN → Evaluates all → Shows comparison plot
```

## Training Individual Algorithm
```bash
python train_baseline.py  # QJC (Q-Learning Jammer)
python train.py           # PPO (Proximal Policy Optimization)
python train_dqn.py       # DQN (Deep Q-Network)
```

## Evaluating Trained Model
```bash
python evaluate.py --algo PPO --dir ./ray_results --no-viz
python evaluate.py --algo DQN --dir ./ray_results_dqn --no-viz
python main.py --no-viz  # Baseline (auto-loads from baseline_q_table/)
```

## Generating Comparison
```bash
python visualization/compare.py
# Creates 3-panel plot: Jamming Success, SINR Impact, Channel Tracking
```

---

# Research-Specific Notes

## Reward Mechanism Design (CRITICAL)
The tracking reward MUST be conditional on power usage:
```python
if (jammer_channel == uav_channel AND jammer_power > 0.01):
    reward_tracking = 0.5
```
**Why:** Prevents "zero-power channel tracking" exploit

## Known Issues & Fixes
1. **DQN Ray Bug:** `ABCMeta` type error in Ray 2.53.0 - Documented in RAPOR.md v1.8.0
2. **Evaluation Bug:** Missing `actions['jammer_0'] = jam_action` - Fixed in evaluate.py
3. **Action Space Mismatch:** Train/eval must use same `flatten_actions` - Fixed via GlobalConfig

## Performance Baselines (Current)
From latest experiments with fixed reward:
- **Baseline (QJC):** 1.02 avg jammed nodes
- **PPO:** 2.79 avg jammed nodes (BEST - 173% improvement)
- **DQN:** 1.99 avg jammed nodes (95% improvement)
