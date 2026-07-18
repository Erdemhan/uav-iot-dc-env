import pickle
import os
import glob
import sys
import json
import optuna
import shutil

# Define dummy function for pickle load
def short_trial_dirname_creator(trial):
    pass

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def verify_trial_iterations(logdir):
    """
    Check if a trial has completed at least 999 training iterations
    by reading its result.json file from disk.
    """
    result_json_path = os.path.join(logdir, "result.json")
    if not os.path.exists(result_json_path):
        return False
        
    try:
        with open(result_json_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 999:
            return False
            
        last_line = json.loads(lines[-1].strip())
        iter_count = last_line.get("training_iteration", 0)
        
        if iter_count >= 999:
            return True
    except:
        pass
        
    return False

def main():
    print("=== REBUILDING STATE FILES FROM DISK TRIAL DIRECTORIES ===")
    
    # Target directory
    run_dir = os.path.join(PROJECT_ROOT, "artifacts", "tune", "tune_ppo_lstm_phase1_2026-07-17_10-29-00")
    optuna_study_dir = os.path.join(run_dir, "tune_results", "optuna_study")
    
    if not os.path.exists(optuna_study_dir):
        print(f"[ERROR] Optuna study directory not found: {optuna_study_dir}")
        sys.exit(1)
        
    # Scan physical folders
    trial_dirs = sorted(glob.glob(os.path.join(optuna_study_dir, "trial_*")))
    print(f"Found {len(trial_dirs)} trial directories on disk.")
    
    completed_trials = []
    ray_trial_data = []
    
    for td in trial_dirs:
        trial_id = os.path.basename(td).replace("trial_", "")
        res_path = os.path.join(td, "result.json")
        params_path = os.path.join(td, "params.json")
        
        if not os.path.exists(res_path) or not os.path.exists(params_path):
            continue
            
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) < 999:
                continue
                
            last_line = json.loads(lines[-1].strip())
            iter_count = last_line.get("training_iteration", 0)
            objective = last_line.get("objective")
            
            if iter_count < 999 or objective is None:
                continue
                
            with open(params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
                
            print(f"  [FOUND] Trial {trial_id}: reached {iter_count} iters, objective = {objective}")
            completed_trials.append((trial_id, params, objective, last_line))
            
            # Construct compliant Ray trial dictionary
            ray_trial = {
                "trial_id": trial_id,
                "status": "TERMINATED",
                "logdir": td,
                "evaluated_params": params,
                "experiment_tag": "",
                "last_result": last_line,
                "last_update_time": 0.0
            }
            # Add wrapped in list for Ray schema liveness
            ray_trial_data.append([json.dumps(ray_trial)])
            
        except Exception as e:
            print(f"    Error parsing {trial_id}: {e}")
            
    print(f"\nReconstructed {len(completed_trials)} completed trials successfully.")
    
    if not completed_trials:
        print("[ERROR] No completed trials found. Cannot rebuild state files.")
        sys.exit(1)
        
    # Recreate Optuna study
    print("\nRebuilding Optuna study state...")
    study = optuna.create_study(study_name="optuna_study", direction="maximize")
    _ot_trials = {}
    _completed_trials = set()
    
    # DYNAMIC CHOICES RESOLUTION:
    arch_choices = set()
    lstm_choices = set()
    seq_choices = set()
    
    for _, params, _, _ in completed_trials:
        if "architecture" in params:
            arch_choices.add(params["architecture"])
        if "lstm_cell_size" in params:
            lstm_choices.add(params["lstm_cell_size"])
        if "max_seq_len" in params:
            seq_choices.add(params["max_seq_len"])
            
    # Default fallback values
    arch_choices.update(["128,256", "256,256", "256,128,128", "128,512,128", "256,256,256", "512,256,256", "512,256", "512,256,256,128"])
    lstm_choices.update([32, 64, 128, 256])
    seq_choices.update([10, 20, 50, 100])
    
    hpo_keys = {"lr", "gamma", "architecture", "lstm_cell_size", "max_seq_len"}
    
    for idx, (trial_id, params, objective, _) in enumerate(completed_trials):
        filtered_params = {k: v for k, v in params.items() if k in hpo_keys}
        
        t = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=objective,
            params=filtered_params,
            distributions={
                "lr": optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True),
                "gamma": optuna.distributions.FloatDistribution(0.8, 0.99),
                "architecture": optuna.distributions.CategoricalDistribution(sorted(list(arch_choices))),
                "lstm_cell_size": optuna.distributions.CategoricalDistribution(sorted(list(lstm_choices))),
                "max_seq_len": optuna.distributions.CategoricalDistribution(sorted(list(seq_choices)))
            }
        )
        study.add_trial(t)
        
    # Map optuna trials to ray IDs
    for idx, ot_trial in enumerate(study.trials):
        ray_tid = completed_trials[idx][0]
        _ot_trials[ray_tid] = ot_trial
        _completed_trials.add(ray_tid)
        
    # 1. Write PKL file (goes inside optuna_study_dir)
    pkl_files = sorted(glob.glob(os.path.join(optuna_study_dir, "searcher-state-*.pkl")))
    pkl_path = pkl_files[-1] if pkl_files else os.path.join(optuna_study_dir, "searcher-state-2026-07-17_10-29-00.pkl")
    
    state_dict = {
        "_ot_study": study,
        "_ot_trials": _ot_trials,
        "_completed_trials": _completed_trials
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(state_dict, f)
    print(f"[PKL REBUILT] {pkl_path}")
    
    # 2. Write JSON experiment state file
    # FIX: Ray Tune looks for experiment_state-*.json directly under run_dir!
    state_files = sorted(glob.glob(os.path.join(run_dir, "experiment_state-*.json")))
    state_path = state_files[-1] if state_files else os.path.join(run_dir, "experiment_state-2026-07-17_10-29-00.json")
    
    experiment_state = {
        "trial_data": ray_trial_data,
        "runner_data": json.dumps({
            "_cached_trial_decisions": {},
            "_queued_trial_decisions": {}
        }),
        "stats": {}
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(experiment_state, f, indent=4)
    print(f"[JSON REBUILT] {state_path}")
    
    # 3. Clear ConcurrencyLimiter JSON
    json_files = sorted(glob.glob(os.path.join(optuna_study_dir, "search_gen_state-*.json")))
    if json_files:
        json_path = json_files[-1]
        try:
            with open(json_path, "rb") as f:
                limiter_data = pickle.load(f)
            limiter_state = limiter_data.get("name:ConcurrencyLimiter", {})
            limiter_state["live_trials"] = set()
            limiter_state["num_unfinished_live_trials"] = 0
            
            exp = limiter_data.get("experiment")
            if exp is not None and hasattr(exp, "spec") and isinstance(exp.spec, dict):
                exp.spec["trial_dirname_creator"] = None
                
            with open(json_path, "wb") as f:
                pickle.dump(limiter_data, f)
            print(f"[LIMITER CLEARED] {json_path}")
        except Exception as e:
            print(f"Failed to clear ConcurrencyLimiter: {e}")
            
    # 4. Copy rebuilt files to /tmp/ray session if path exists
    temp_run_dir = "/tmp/ray/session_2026-07-17_10-05-50_996511_3860/artifacts/2026-07-17_10-29-00"
    if os.path.exists(temp_run_dir):
        # Temp ray session structure has pkl/json inside driver_artifacts, and experiment_state directly under artifacts/
        temp_driver_dir = os.path.join(temp_run_dir, "optuna_study", "driver_artifacts")
        if os.path.exists(temp_driver_dir):
            shutil.copy2(pkl_path, os.path.join(temp_driver_dir, os.path.basename(pkl_path)))
            if json_files:
                shutil.copy2(json_path, os.path.join(temp_driver_dir, os.path.basename(json_path)))
        if os.path.exists(temp_run_dir):
            shutil.copy2(state_path, os.path.join(temp_run_dir, os.path.basename(state_path)))
        print("[TEMP COPIED] Rebuilt files copied to temp ray session directory.")
            
    print("\n=== REBUILD SUCCESSFUL! ===")
    print("Tum veri tabaniniz diskteki trial verilerinizden sifirdan ve kusursuz olarak insa edildi.")
    print("Artik '--num-samples 100' ile baslatabilirsiniz.")

if __name__ == "__main__":
    main()
