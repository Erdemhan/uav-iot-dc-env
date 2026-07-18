import pickle
import os
import glob
import sys

# Define dummy function at module level so pickle.dump can resolve it on __main__
def short_trial_dirname_creator(trial):
    pass

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

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
print(f"Loading search generator state: {state_file}...")

with open(state_file, "rb") as f:
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

# Save back to disk
print("\nSaving patched search generator state to disk...")
with open(state_file, "wb") as f:
    pickle.dump(data, f)
print("[SUCCESS] search_gen_state successfully patched!")
