# Agent Rules and Constraints

To ensure consistency in research and prevent configuration conflicts, the AI agent must strictly follow these rules:

## 1. Parameters & Configuration Constraints
* **NO Unauthorized Parameter Changes:** Do NOT change the action space, state/observation space, or scenario parameters (e.g., `NUM_UAVS`, `NUM_NODES`, reward weights, etc.) in `confs/env_config.py` or other files without explicitly asking the user and receiving their approval.
* **No Scenario Modification:** Do not switch between single-UAV and multi-UAV configurations, or change frequencies, without explicit user instruction.

## 2. Execution Constraints
* **NO Unauthorized Experiments/Training:** Do NOT run training, experiments, or full pipelines (e.g., `run_experiments.py`) without the user's explicit approval. 
* **Allowed Test Runs:** Running evaluation scripts on existing checkpoints (`evaluate_paper_robustness.py` or `evaluate.py`) is allowed for validation, provided they do not overwrite trained models.
