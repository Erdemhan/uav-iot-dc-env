---
name: experiment_tracker
description: Monitors and reports the results of parameter search experiments by reading metadata.json and evaluation logs from artifacts.
---

# Experiment Tracker & Reporter Skill

You have triggered the `experiment_tracker` skill because a training experiment run has completed or needs parameter/performance analysis.

## Objectives
Extract hyperparameters and evaluation results from experiment trial artifacts, summarize them in a structured comparison report, and log interpretations.

## Guidelines & Rules

1. **Locate Latest Experiment**:
   - Locate the latest run folder inside the `artifacts/` directory (formatted as `artifacts/YYYY-MM-DD_HH-MM-SS/`).

2. **Extract Metadata & Hyperparameters**:
   - Read the `{run_dir}/metadata.json` file.
   - Extract key parameters for the trial:
     - Global configurations (`TRAIN_ITERATIONS`, `TRAIN_BATCH_SIZE`, `RANDOM_SEED`, etc.)
     - Model configs (PPO: LR, Network layers. DQN: Double/Dueling parameters, Update Freq. LSTM: Cell Size, Seq Length).
     - Env/UAV configuration settings.

3. **Extract Performance Results**:
   - Check evaluation folders (e.g. `{run_dir}/ppo/evaluation/`, `{run_dir}/dqn/evaluation/`, etc.) or read output metrics (such as average jammed nodes, SINR levels, channel tracking rewards).
   - If available, look for comparison metrics generated in the `{run_dir}/comparison/` folder.

4. **Format Structured Trial Report**:
   - Create a markdown section comparing the parameters and performance.
   - Example table format:
     | Algorithm | Key Parameters (LR, Gamma, Layer, etc.) | Avg Jammed Nodes | SINR Impact | Tracking Success |
     |---|---|---|---|---|
     | **Baseline (QJC)** | `TAU_0=1e-4`, `GAMMA=0.9` | - | - | - |
     | **PPO** | `LR=1e-4`, `Layers=[256, 256]` | - | - | - |
     | **DQN** | `LR=1e-4`, `Double=True` | - | - | - |
     | **PPO-LSTM** | `LSTM_Cell=256`, `Seq_Len=20` | - | - | - |

5. **Generate Interpretations & Insights**:
   - Compare performance across different runs/trials.
   - Draft technical explanations for the differences (e.g., "LSTM cell size 256 improved temporal pattern matching compared to standard PPO because...").
   - Suggest next parameter variations to test (e.g. learning rate decay, target network update frequency tweaks).
