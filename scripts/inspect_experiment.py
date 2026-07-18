import pickle
import os
import glob
import sys

# Standalone dummy function definition in __main__ namespace to satisfy pickle load
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
target_dir = os.path.join(run_dir, "tune_results", "optuna_study")
target_file = os.path.join(target_dir, "search_gen_state-2026-07-17_10-29-00.json")

with open(target_file, "rb") as f:
    data = pickle.load(f)

exp = data.get("experiment")
print("=== EXPERIMENT ATTRIBUTES ===")
if exp is not None:
    print(f"Class: {type(exp)}")
    for k, v in exp.__dict__.items():
        print(f"  {k}: type={type(v)}")
        if isinstance(v, dict):
            print(f"    Keys: {list(v.keys())}")
            for nk, nv in v.items():
                print(f"      {nk}: type={type(nv)}")
else:
    print("Experiment is None!")
