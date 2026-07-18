import os
import json
import glob
import sys

def main():
    # Dynamic project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to PPO-LSTM run directory on WSL (can be passed or auto-detected)
    # Defaulting to the latest PPO-LSTM folder in artifacts/tune
    tune_dir = os.path.join(PROJECT_ROOT, "artifacts", "tune")
    if not os.path.exists(tune_dir):
        # Fallback to artifacts/
        tune_dir = os.path.join(PROJECT_ROOT, "artifacts")
        
    ppo_lstm_runs = sorted(glob.glob(os.path.join(tune_dir, "tune_ppo_lstm_phase1_2026-07-*")), reverse=True)
    if not ppo_lstm_runs:
        print("[ERROR] No PPO-LSTM run folder found to patch.")
        return
        
    run_dir = ppo_lstm_runs[0]
    print(f"Target run directory to patch: {run_dir}")
    
    tune_results_dir = os.path.join(run_dir, "tune_results", "optuna_study")
    state_files = glob.glob(os.path.join(tune_results_dir, "experiment_state-*.json"))
    if not state_files:
        print("[ERROR] No experiment_state-*.json file found.")
        return
        
    state_file = state_files[0]
    print(f"Found experiment state file: {state_file}")
    
    with open(state_file, "r", encoding="utf-8") as f:
        state_data = json.load(f)
        
    trial_list = state_data.get("trial_data", [])
    running_count = 0
    patched_count = 0
    
    new_trial_list = []
    for item in trial_list:
        if isinstance(item, list) and len(item) > 0:
            elem = item[0]
            if isinstance(elem, str):
                try:
                    elem_dict = json.loads(elem)
                    if isinstance(elem_dict, dict) and elem_dict.get("status") == "RUNNING":
                        # Patch status from RUNNING to TERMINATED so Ray doesn't auto-resume them
                        # and exceed concurrency limit
                        elem_dict["status"] = "TERMINATED"
                        item[0] = json.dumps(elem_dict)
                        patched_count += 1
                except Exception as e:
                    pass
        new_trial_list.append(item)
        
    state_data["trial_data"] = new_trial_list
    
    # Save the patched state
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=4)
        
    print(f"\n[SUCCESS] Patched {patched_count} RUNNING trials to TERMINATED in experiment state.")
    print("You can now safely resume PPO-LSTM, and it will strictly run with exactly 4 concurrent trials!")

if __name__ == "__main__":
    main()
