import pickle
import os
import glob
import sys
import shutil

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import the real function to satisfy pickle's identity check during serialization
from scripts.tune_models import short_trial_dirname_creator

# 1. Target files
tune_dir = os.path.join(PROJECT_ROOT, "artifacts", "tune")
ppo_lstm_runs = sorted(glob.glob(os.path.join(tune_dir, "tune_ppo_lstm_phase1_2026-07-*")), reverse=True)
if not ppo_lstm_runs:
    print("[ERROR] No PPO-LSTM run folder found.")
    sys.exit(1)

run_dir = ppo_lstm_runs[0]
target_dir = os.path.join(run_dir, "tune_results", "optuna_study")
target_file = os.path.join(target_dir, "search_gen_state-2026-07-17_10-29-00.json")

# 2. Find backup in /tmp
session_pattern = "/tmp/ray/session_2026-07-17_*/artifacts/2026-07-17_10-29-00/optuna_study/driver_artifacts/search_gen_state-2026-07-17_10-29-00.json"
backup_files = glob.glob(session_pattern)

if not backup_files:
    backup_files = glob.glob("/tmp/ray/session_*/artifacts/*/optuna_study/driver_artifacts/search_gen_state-2026-07-17_10-29-00.json")

if not backup_files:
    print("[ERROR] Could not find any backup of search_gen_state in /tmp/ray/session_*")
    sys.exit(1)

backup_file = backup_files[0]
print(f"Found backup file at: {backup_file} ({os.path.getsize(backup_file)} bytes)")

# 3. Restore backup to target directory
print(f"Restoring backup to: {target_file}...")
shutil.copy2(backup_file, target_file)

# 4. Load restored file
print("Loading restored state file...")
with open(target_file, "rb") as f:
    data = pickle.load(f)

limiter_state = data.get("name:ConcurrencyLimiter")
if limiter_state is None:
    print("[ERROR] Could not find 'name:ConcurrencyLimiter' in the state.")
    sys.exit(1)

print("\nBefore Patch:")
print(f"  live_trials: {limiter_state.get('live_trials')}")
print(f"  num_unfinished_live_trials: {limiter_state.get('num_unfinished_live_trials')}")

# Apply patch
limiter_state["live_trials"] = "set()"
limiter_state["num_unfinished_live_trials"] = 0

print("\nAfter Patch:")
print(f"  live_trials: {limiter_state.get('live_trials')}")
print(f"  num_unfinished_live_trials: {limiter_state.get('num_unfinished_live_trials')}")

# 5. Safely save using temporary file (atomic write)
temp_file = target_file + ".tmp"
print(f"\nWriting patched state to temp file: {temp_file}...")
try:
    with open(temp_file, "wb") as f:
        pickle.dump(data, f)
    # Rename temp file to target file (atomic overwrite)
    shutil.move(temp_file, target_file)
    print("[SUCCESS] search_gen_state successfully restored and patched!")
except Exception as e:
    print(f"[ERROR] Failed during pickling: {e}")
    if os.path.exists(temp_file):
        os.remove(temp_file)
