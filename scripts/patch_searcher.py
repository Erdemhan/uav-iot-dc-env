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

# Extract Optuna study
study = state_dict.get("_ot_study")
if study is None:
    print("[ERROR] Could not find '_ot_study' in the searcher state.")
    sys.exit(1)

print(f"Loaded study '{study.study_name}' with {len(study.trials)} trials.")

# Find trials in RUNNING state
running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
print(f"Found {len(running_trials)} trials in RUNNING state in Optuna study.")

if running_trials:
    # Set their state to FAIL in Optuna storage
    for t in running_trials:
        print(f"  Patching Trial #{t.number} (ID: {t._trial_id}) state from RUNNING to FAIL...")
        try:
            # Optuna 3.0+ API
            study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
        except AttributeError:
            # Optuna 2.0+ API fallback
            study._storage.set_trial_state(t._trial_id, optuna.trial.TrialState.FAIL)
    
    # Save the updated state back to the pickle file
    print("Saving patched searcher state back to disk...")
    with open(path, "wb") as f:
        pickle.dump(state_dict, f)
    print("[SUCCESS] Searcher state patched successfully!")
else:
    print("No RUNNING trials found in the study. No patching needed.")
