---
name: experiment_tracker
description: Monitors and reports the results of parameter search experiments by reading metadata.json and evaluation logs from artifacts.
---

# Experiment Tracker & Reporter Skill

You have triggered the `experiment_tracker` skill because a training experiment run has completed or needs parameter/performance analysis.

## Objectives
Extract hyperparameters and evaluation results from experiment trial artifacts, summarize them in a structured comparison report, and log interpretations.

## Guidelines & Rules

0. **Language Requirement**:
   - The generated report and all interpretations must be written in **Turkish**.

1. **Locate Latest Experiment**:
   - Locate the latest run folder inside the `artifacts/` directory (formatted as `artifacts/YYYY-MM-DD_HH-MM-SS/`).

2. **Extract Metadata & Hyperparameters**:
   - Read the `{run_dir}/metadata.json` file.
   - Extract key parameters for the trial:
     - Global configurations (`TRAIN_ITERATIONS`, `TRAIN_BATCH_SIZE`, `RANDOM_SEED`, etc.)
     - Model configs (PPO: LR, Network layers. DQN: Double/Dueling parameters, Update Freq. LSTM: Cell Size, Seq Length).
     - Env/UAV configuration settings.

3. **Extract Performance & Robustness Results**:
   - Check evaluation folders (e.g. `{run_dir}/ppo/evaluation/`, `{run_dir}/dqn/evaluation/`, etc.) or read output metrics.
   - Read `{run_dir}/comparison/robustness_results_30seeds.json` to extract metrics calculated over 30 random seeds:
     - **JSR (Jamming Success Rate)**: mean ± std
     - **Tracking Accuracy**: mean ± std
     - **Power Consumption**: mean ± std
     - **SINR**: mean ± std

4. **Format and Save Structured Trial Report**:
   - Create a markdown section comparing the parameters and performance.
   - Save this report directly inside the corresponding experiment trial directory as `{run_dir}/comparison/experiment_report.md`.
   - Example table format:
     | Algorithm | Key Parameters (LR, Gamma, Layer, etc.) | JSR (30 Seeds) | Tracking Acc (30 Seeds) | Avg Power | Avg SINR |
     |---|---|---|---|---|---|
     | **Baseline (QJC)** | `TAU_0=1e-4`, `GAMMA=0.9` | - | - | - | - |
     | **PPO** | `LR=1e-4`, `Layers=[256, 256]` | - | - | - | - |
     | **DQN** | `LR=1e-4`, `Double=True` | - | - | - | - |
     | **PPO-LSTM** | `LSTM_Cell=256`, `Seq_Len=20` | - | - | - | - |

5. **Generate Interpretations & Insights**:
   - Compare performance across different runs/trials.
   - Draft technical explanations for the differences (e.g., "LSTM cell size 256 improved temporal pattern matching compared to standard PPO because...").
   - Comment specifically on algorithm **robustness** and standard deviation across seeds.
   - Suggest next parameter variations to test (e.g. learning rate decay, target network update frequency tweaks).
