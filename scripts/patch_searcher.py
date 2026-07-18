import pickle
import os
import sys
import optuna

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Path to the PPO-LSTM searcher state file
path = os.path.join(PROJECT_ROOT, "artifacts", "tune", "tune_ppo_lstm_phase1_2026-07-17_10-29-00", "tune_results", "optuna_study", "searcher-state-2026-07-17_10-29-00.pkl")

if not os.path.exists(path):
    # Try alternate path
    path = os.path.join(PROJECT_ROOT, "artifacts", "tune_ppo_lstm_phase1_2026-07-17_10-29-00", "tune_results", "optuna_study", "searcher-state-2026-07-17_10-29-00.pkl")

if not os.path.exists(path):
    print(f"[ERROR] Pickle file not found at: {path}")
    sys.exit(1)

print(f"Loading searcher state: {path}...")
with open(path, "rb") as f:
    state_dict = pickle.load(f)

# 1. Update Optuna Study trial states
study = state_dict.get("_ot_study")
if study is None:
    print("[ERROR] Could not find '_ot_study' in the searcher state.")
    sys.exit(1)

print(f"Loaded study '{study.study_name}' with {len(study.trials)} trials.")
running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
print(f"Found {len(running_trials)} trials in RUNNING state in Optuna study.")

if running_trials:
    for t in running_trials:
        print(f"  Patching Trial #{t.number} (ID: {t._trial_id}) state from RUNNING to FAIL...")
        try:
            study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
        except AttributeError:
            study._storage.set_trial_state(t._trial_id, optuna.trial.TrialState.FAIL)

# 2. Update Completed Trials Set to unblock Ray's concurrency limiter
completed_set = state_dict.get("_completed_trials", set())
ot_trials_dict = state_dict.get("_ot_trials", {})

# Calculate active trial IDs in Ray's searcher wrapper
active_ids = set(ot_trials_dict.keys()) - completed_set
print(f"\nCurrently registered active trials in searcher: {list(active_ids)}")

if active_ids:
    print(f"Patching searcher state: adding {len(active_ids)} active trial IDs to '_completed_trials'...")
    completed_set.update(active_ids)
    state_dict["_completed_trials"] = completed_set
    
    # Save the updated state back to the pickle file
    print("Saving patched searcher state back to disk...")
    with open(path, "wb") as f:
        pickle.dump(state_dict, f)
    print("[SUCCESS] Searcher state patched successfully!")
else:
    print("No active trials found in the completed trials set. No patching needed.")
