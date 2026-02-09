# Simulation Workflow

This document explains the working logic of the `UAV_IoT_Sim` project step by step.

## Project Structure
```
uav-iot-dc-env/
├── confs/              # Configuration Files
│   ├── config.py       # System Configuration (Physics, Delay)
│   ├── env_config.py   # Environment & Scenario Config (Nodes, Power, Speed)
│   ├── model_config.py # Centralized ML Config (GlobalConfig, PPOConfig, DQNConfig, QJCConfig)
├── core/               # Core modules
│   ├── physics.py      # Physics engine
│   └── logger.py       # Logging system
├── simulation/         # Simulation environment
│   ├── pettingzoo_env.py # PettingZoo ParallelEnv
│   ├── controllers.py    # Rule-based controllers (UAV)
│   └── entities.py       # Entity classes
├── scripts/            # Execution Scripts
│   ├── main.py         # Single-Run Visualizer
│   ├── train.py        # PPO Training
│   ├── train_dqn.py    # DQN Training
│   ├── train_ppo_lstm.py # PPO-LSTM Training (Recurrent)
│   ├── run_experiments.py # Parallel Automation (Main Entry)
│   ├── evaluate.py     # Unified Evaluation
│   └── evaluate_paper_robustness.py # 30-Seed Robustness Test
├── visualization/      # Visualization
│   ├── visualization.py # Runtime visualization
│   ├── visualizer.py   # Analysis and reporting
│   └── compare.py      # Multi-Algo Comparison Plotter
├── artifacts/          # Experiment Artifacts (Timestamped)
│   └── 2026-02-08_.../ # Individual Run Results
├── logs/               # Legacy Simulation outputs
├── README.md           # Workflow documentation
└── RAPOR.md            # Technical report (Turkish)
```

## 1. Initialization - `main.py`
When the simulation is started with `python scripts/main.py`, the following happens sequentially:

1.  **Logger Setup (`SimulationLogger`)**:
    *   A new folder with the current date/time is created under `logs/` (e.g., `EXP_20260201_200542`).
    *   Simulation parameters are read from `confs/config.py` and `confs/env_config.py`, then saved as `config.json` in this folder.
    
2.  **Environment Setup (`UAV_IoT_PZ_Env`)**:
    *   **Multi-Agent Structure**: The environment is initialized as a PettingZoo `ParallelEnv`.
    *   **Agents**:
        *   `uav_0`: Controlled via velocity commands (currently by `UAVRuleBasedController`).
        *   `jammer_0`: Controlled via power commands.
        *   `node_0`..`node_N`: Passive IoT nodes.
    *   `UAVRuleBasedController`: A controller is instantiated to manage the UAV's waypoint navigation logic.
    
3.  **Visualization (`Visualization`)**:
    *   A Matplotlib window opens, and interactive mode (`plt.ion()`) is enabled.

## 2. Simulation Loop
The simulation runs for a specified number of steps (Default: 100). In each step (`env.step(action)`), the following operations are performed:

### A. Action Selection
*   **UAV (`uav_0`)**: The `UAVRuleBasedController` manages navigation. It moves towards the next waypoint and, upon arrival, **hovers** for 5 seconds (1 step) to collect data before moving to the next node.
*   **Jammer (`jammer_0`)**: A random jamming power (or RL-based action) is selected.
*   **Nodes**: Passive (No-Op).
*   All actions are passed to `env.step(actions)` dictionary.

### B. Physical Calculations (`core/physics.py`)
1.  **UAV Movement**:
    *   The UAV is moved in a circular trajectory (r=200m).
    *   New position and velocity vectors are updated.
2.  **Channel and Communication**:
    *   **UAV -> Node** and **Jammer -> Node** distances are calculated.
    *   Path losses are found using `physics.calculate_path_loss`.
    *   **SINR** (Signal/Noise+Jamming) is calculated for each node.
    *   Instantaneous **Data Rate** is found using the Shannon equation.

### C. State Updates (`simulation/entities.py`)
1.  **Connection Check**: If SINR is below a certain threshold (e.g., 1.0), the node is considered "Jammed".
2.  **Age of Information (AoI)**:
    *   If connected: AoI = 0 (Reset).
    *   If not connected: AoI += time elapsed (Stale).
3.  **Energy Consumption**:
    *   **UAV**: Aerodynamic power consumption based on flight speed is calculated.
    *   **IoT Node**: Costs for data sending and encryption are calculated.
    *   **Jammer**: Power is calculated using the PA Efficiency model:
        $$ P_{sys} = \frac{P_{jam}}{\eta_{PA}(f)} + P_{circuit} $$
        (Efficiency: %50 @ 2.4GHz, %19 @ 5.8GHz). This rewards the agent for using efficient frequencies.

### D. Logging (`core/logger.py`)
At the end of each step, all critical data is buffered:
*   Step No
*   UAV and Attacker Positions
*   For each Node: SINR, AoI, Energy Consumed
*   Attacker's current power

### E. Visualization (`render`)
*   Points on the map are updated.
*   **Jamming Contour**: A dynamic red contour is drawn where SINR < 0 dB, showing the real-time effective jamming zone.
*   Instantaneous step and power information is written to the screen.

## 3. Reward Mechanisms

The jammer's training is driven by carefully designed reward functions that balance jamming effectiveness with energy efficiency.

### A. Baseline (QJC Algorithm)
The Q-Learning approach uses a discrete action-reward table:
*   **State**: Current channel (0, 1, 2)
*   **Action**: Power level (0-9)
*   **Reward**: Number of jammed nodes × 10 - energy cost × 0.1

### B. PPO, PPO-LSTM & DQN (Deep RL)
Both algorithms share the same reward structure with three components:

#### 1. Jamming Success Reward (Sparse, High)
```
reward_success = jammed_node_count × 10
```
*   **Purpose**: Primary objective - maximize jamming effectiveness
*   **Range**: 0 to 50 (for 5 nodes)
*   **Type**: Sparse reward (only when jamming occurs)

#### 2. Channel Tracking Reward (Dense, Low)
```python
if (jammer_channel == uav_channel AND jammer_power > 0.01):
    reward_tracking = 0.5
else:
    reward_tracking = 0.0
```
*   **Purpose**: Guide the jammer to track UAV's frequency hopping
*   **Condition**: CRITICAL - Only given when actually using power (prevents exploitation)
*   **Type**: Dense guidance signal

#### 3. Energy Cost Penalty
```
reward_energy = -jammer_power_consumption × 0.1
```
*   **Purpose**: Encourage energy-efficient jamming strategies
*   **Range**: 0 to -0.01W (typical)

#### Total Reward
```
total_reward = reward_success + reward_tracking - reward_energy
```

### C. Reward Design Rationale
*   **Power Threshold Check**: The tracking reward requires `power > 0.01W` to prevent agents from learning a "zero-power channel tracking" exploit.
*   **Scaling Balance**: Success reward (10×) dominates tracking (0.5), ensuring jamming is the primary objective.
*   **Energy Trade-off**: Cost penalty (0.1×) is small enough to not deter jamming but large enough to prefer efficient strategies.

## 4. Termination
When the loop ends or is stopped by the user:
1.  **Data Recording**: All data in the buffer is written to the `history.csv` file.
2.  Window is closed.
3.  The location of the log files is printed to the screen.

## 4. Analysis and Visualization - `visualization/visualizer.py`
To graph the results after the simulation ends:
```bash
python visualization/visualizer.py 
# Note: Usually called automatically by main.py
```
This command finds the latest experiment folder and generates analysis graphs (Trajectory, Metrics, Communication Stats).

### RLLib Bug Fix (Critical)
During development, a `TypeError` was discovered in Ray 2.53.0's DQN implementation. 
*   **Issue:** `ABCMeta` type was not iterable in `_create_local_replay_buffer_if_necessary`.
*   **Fix:** A patch was applied to the local library.
*   **PR:** A Pull Request has been submitted to the `ray-project/ray` repository.
*   **Documentation:** Detailed the issue and fix in `pr.md`.

## 5. Configuration Management (`confs/model_config.py`)

The project uses a centralized configuration system for reproducibility and easy hyperparameter tuning:

### Configuration Classes:

*   **GlobalConfig**: Shared parameters across all algorithms
    *   `RANDOM_SEED`: Random seed for reproducibility (42)
    *   `FLATTEN_ACTIONS`: Action space flattening for DQN compatibility (True)
    *   `TRAIN_ITERATIONS`: Training iterations for all RL algorithms (20)

*   **QJCConfig**: Baseline Q-Learning parameters
    *   Learning rate (TAU_0), discount factor (GAMMA), softmax temperature (TEMP_XI)
    *   Training episodes, save path, max power level

*   **PPOConfig**: PPO training parameters
    *   Learning rate (LR), gamma, batch size, rollout fragment length
    *   Model architecture (`FCNET_HIDDENS = [256, 256]`)
    *   GPU settings (`USE_GPU = True`)

*   **PPOLSTMConfig**: PPO with LSTM parameters (Recurrent Policy)
    *   **Architecture**: Adds an LSTM cell (size 256) before the Action Head.
    *   **Purpose**: Solves "Partial Observability" (POMDP). The agent remembers past observations (e.g., UAV's last known frequency) to predict future states even when the UAV is efficient hopping.
    *   **Config**: `USE_LSTM = True`, `MAX_SEQ_LEN = 20`.

*   **DQNConfig**: DQN training parameters
    *   Learning rate, gamma, batch size, target network update frequency
    *   Replay buffer capacity, Double-Q, Dueling settings
    *   Model architecture and GPU settings

### Benefits:
*   **Single Source of Truth**: All hyperparameters in one file
*   **Easy Experimentation**: Change seed or hyperparameters with one edit
*   **Reproducibility**: Guaranteed consistent random initialization
*   **No Code Duplication**: Eliminates hardcoded values across scripts

### API Stack Fairness:
For fair algorithmic comparison, both PPO and DQN use the **Old API Stack**:
*   **PPO**: Configured with `api_stack(enable_rl_module_and_learner=False)`
*   **DQN**: Naturally uses Old API Stack
*   **Benefit**: Identical GPU reporting, resource allocation, and execution behavior

## 6. GPU Setup (CUDA Support)

### Requirements:
*   NVIDIA GPU (Tested on RTX 3080)
*   CUDA 12.1+ installed

### Installation:
If PyTorch doesn't detect your GPU:
```bash
# Uninstall CPU-only version
pip uninstall torch torchvision torchaudio -y

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verification:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

Expected output: `CUDA Available: True`

**Note**: GPU acceleration provides ~5-10x speedup during training.

## 7. Automated Parallel Comparison (`run_experiments.py`)
To run the full scientific comparison pipeline (Official Baseline vs PPO vs DQN) in **PARALLEL**:
```bash
python scripts/run_experiments.py
```
This script automates:
1.  **Parallel Execution**: Launches Baseline, PPO, DQN, and PPO-LSTM training simultaneously (4x faster).
2.  **Real-Time Monitoring**: Displays color-coded progress bars and iteration counts in the terminal.
3.  **Artifact Management**: Creates a timestamped folder in `artifacts/` containing all models, logs, and plots.
4.  **Auto-Report**: Generates `comparison_result.png` showing success rates and energy efficiency.

### Command-Line Arguments:
*   `--debug`: Enable verbose output (shows all subprocess logs and errors).
*   `--ui <seconds>`: Set custom terminal update interval (Default: 3s).

Example:
```bash
python scripts/run_experiments.py --debug --ui 1
```

## 8. Evaluation & Visualization (`evaluate.py`)
To visualize the behavior of the **latest trained** model:
```bash
python scripts/evaluate.py
```
This script runs a single interactive episode, allowing you to observe the learned Markov-channel-switching prediction logic in real-time.

## 9. Reproducing Paper Results (Robustness Analysis)

To generate the specific robust statistics (30-Seed Analysis) used in the paper:
```bash
python scripts/evaluate_paper_robustness.py
```
This script automates the full scientific validation process:
1.  **Iterative Evaluation**: Runs the trained models over 30 distinct random seeds (Range: 100-129).
2.  **Data Collection**: Aggregates JSR, Tracking Accuracy, Power, and SINR metrics.
3.  **Statistical Output**: Saves mean/std statistics.
4.  **Plot Generation**: Creates graph (Bar Chart with Error Bars).

**Note:** This process ensures the findings are statistically significant and not result of a lucky seed.


