import pickle
import os
import glob
import sys
import json

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tune_dir = os.path.join(PROJECT_ROOT, "artifacts", "tune")
ppo_lstm_runs = sorted(glob.glob(os.path.join(tune_dir, "tune_ppo_lstm_phase1_2026-07-*")), reverse=True)
if not ppo_lstm_runs:
    print("[ERROR] No PPO-LSTM run folder found.")
    sys.exit(1)

run_dir = ppo_lstm_runs[0]
tune_results_dir = os.path.join(run_dir, "tune_results", "optuna_study")
state_files = glob.glob(os.path.join(tune_results_dir, "search_gen_state-*.json"))
if not state_files:
    print("[ERROR] No search_gen_state-*.json file found.")
    sys.exit(1)

state_file = state_files[0]
print(f"Reading search generator state file: {state_file}...\n")

try:
    with open(state_file, "rb") as f:
        data = pickle.load(f)
    
    limiter_state = data.get("name:ConcurrencyLimiter", {})
    print("=== ConcurrencyLimiter NESTED STATE ===")
    print(f"Limiter state type: {type(limiter_state)}")
    if isinstance(limiter_state, dict):
        print(json.dumps(limiter_state, indent=4, default=str))
except Exception as e:
    print(f"[ERROR] Failed to print ConcurrencyLimiter state: {e}")
