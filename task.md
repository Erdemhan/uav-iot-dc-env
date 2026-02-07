# Configuration Refactoring

Separating high-level scenario and environment parameters from low-level physical constants.

- [x] **Create `core/env_config.py`**
    - [x] Define `EnvConfig` class.
    - [x] Add scenario parameters: `NUM_NODES`, `AREA_SIZE`.
    - [x] Add simulation stepping parameters: `MAX_STEPS`, `STEP_TIME`, `UAV_START_X`, `UAV_START_Y`, `UAV_START_Z`.
    - [x] Add dynamic parameters: `UAV_SPEED`, `UAV_RADIUS` (currently hardcoded in step).
    - [x] Add Attacker parameters: `ATTACKER_X`, `ATTACKER_Y`, `MAX_JAM_POWER`.

- [x] **Update `core/config.py`**
    - [x] Remove `NUM_NODES`, `AREA_SIZE`.
    - [x] Keep physical/system constants.

- [x] **Refactor `simulation/environment.py`**
    - [x] Import `EnvConfig`.
    - [x] Replace hardcoded values (`max_steps=100`, `dt=5.0`, `radius=200`, `speed=10.0`) with `EnvConfig` values.
    - [x] Use `EnvConfig` for area size and node count.

- [x] **Update Dependent Modules**
    - [x] `simulation/entities.py` (If it uses removed keys).
    - [x] `visualization/visualization.py` (Update `AREA_SIZE` import).
    - [x] `visualization/visualizer.py` (Update `AREA_SIZE` import).
    - [x] `main.py` (Pass both configs to Logger).

- [x] **Smart Threat & RLLib Integration**
    - [x] **Physical Layer & Config**
        - [x] Update `config.py` with Channels, PA Efficiency, and Power constants. <!-- id: 37 -->
        - [x] Update `physics.py` with `calculate_total_jammer_power`. <!-- id: 38 -->
    - [x] **Intelligence & Logic**
        - [x] Update `SmartAttacker` in `entities.py` with QJC (Q-Table, Softmax). <!-- id: 39 -->
        - [x] Update `UAVRuleBasedController` with Markov Channel Switching. <!-- id: 40 -->
    - [x] **Environment Upgrade**
        - [x] Update `pettingzoo_env.py` for Multi-Channel Actions/Obs. <!-- id: 41 -->
        - [x] Implement Frequency-Aware Step Logic (Interference only on same channel). <!-- id: 42 -->
    - [x] **RLLib Training Setup**
        - [x] Create `train.py` with Ray RLLib configuration. <!-- id: 43 -->
    - [x] **Documentation & Viz**
        - [x] Update `visualizer.py` for comparative analysis (Channel Usage). <!-- id: 44 -->
        - [x] Update `README.md` and `RAPOR.md` with references. <!-- id: 45 -->

- [ ] **Reward Engineering & Optimization**
    - [x] **Reward Shaping**
        - [x] Add "Tracking Reward" (Dense reward for channel matching).
        - [x] Normalize Observation Space (Coordinates scaled to [0,1]).
    - [ ] **Hyperparameter Tuning**
        - [x] Increase Training Iterations (20 -> 60).
        - [ ] Tune Entropy Coefficient (if exploration is low).
    - [x] **Debugging & Dependency Fixes**
        - [x] Fix Ray RLLib Dependencies (`dm-tree`, `tensorboardX`).
        - [x] Fix Recursive Checkpoint Loading in `evaluate.py`.
        - [x] Update `train.py` / `evaluate.py` for Ray New API Stack.
        - [x] Fix Missing Node Positions in `trajectory.png`.

- [ ] **Comparison & Scientific Validation**
    - [x] **Fairness Upgrade (Sensing Mode)**
        - [x] Convert Observation: Coordinates (God View) -> Distance/RSSI (Realistic).
        - [x] Retrain PPO with Sensing Data.
    - [x] **Algorithmic Comparison**
        - [x] Create `train_baseline.py` for Pre-training Q-Table.
        - [ ] Create `train_dqn.py` for DQN Algorithm.
        - [x] Create `run_experiments.py` for Automation.
        - [x] Create `visualization/compare.py` for Reporting.
