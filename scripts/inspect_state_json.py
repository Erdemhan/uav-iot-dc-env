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
print(f"Reading state file: {state_file}...")

with open(state_file, "r", encoding="utf-8") as f:
    state_data = json.load(f)

# 1. Search trial_data for any remaining RUNNING or PENDING trials
trial_list = state_data.get("trial_data", [])
print(f"Total trials in trial_data: {len(trial_list)}")

status_counts = {}
for idx, item in enumerate(trial_list):
    if isinstance(item, list) and len(item) > 0:
        elem = item[0]
        if isinstance(elem, str):
            try:
                elem_dict = json.loads(elem)
                status = elem_dict.get("status")
                status_counts[status] = status_counts.get(status, 0) + 1
                if status in ["RUNNING", "PENDING"]:
                    print(f"  Trial {idx} (ID: {elem_dict.get('trial_id')}) has status: {status}")
            except:
                pass

print(f"Trial status counts: {status_counts}")

# 2. Search the rest of the JSON (like runner_data) for any occurrences of "RUNNING"
# Convert runner_data to string and look for keywords
runner_data_str = json.dumps(state_data.get("runner_data", {}))
print(f"\nSearching runner_data (length={len(runner_data_str)})...")
if "RUNNING" in runner_data_str:
    print("  [ALERT] Found 'RUNNING' in runner_data!")
    # Find context of 'RUNNING'
    idx = 0
    while True:
        idx = runner_data_str.find("RUNNING", idx)
        if idx == -1:
            break
        start = max(0, idx - 100)
        end = min(len(runner_data_str), idx + 100)
        print(f"    Context: ... {runner_data_str[start:end]} ...")
        idx += len("RUNNING")
else:
    print("  No 'RUNNING' found in runner_data.")
