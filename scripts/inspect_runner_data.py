import json
import os
import glob
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Find experiment state file
tune_dir = os.path.join(PROJECT_ROOT, "artifacts", "tune")
ppo_lstm_runs = sorted(glob.glob(os.path.join(tune_dir, "tune_ppo_lstm_phase1_2026-07-*")), reverse=True)
if not ppo_lstm_runs:
    print("[ERROR] No PPO-LSTM run folder found.")
    sys.exit(1)

run_dir = ppo_lstm_runs[0]
tune_results_dir = os.path.join(run_dir, "tune_results", "optuna_study")
state_files = glob.glob(os.path.join(tune_results_dir, "experiment_state-*.json"))
if not state_files:
    print("[ERROR] No experiment_state-*.json file found.")
    sys.exit(1)

state_file = state_files[0]
with open(state_file, "r", encoding="utf-8") as f:
    state_data = json.load(f)

runner_data = state_data.get("runner_data", {})
print("=== RUNNER DATA CONTENT ===")
print(json.dumps(runner_data, indent=4))
