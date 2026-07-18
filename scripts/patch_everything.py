import pickle
import os
import glob
import sys
import shutil
import json
import optuna

# Standalone dummy function definition in __main__ namespace to satisfy pickle load/dump
def short_trial_dirname_creator(trial):
    pass

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def patch_searcher_pkl(path):
    print(f"\n[PKL] Processing searcher state file: {path}...")
    try:
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
            
        # 1. Patch Optuna study
        study = state_dict.get("_ot_study")
        if study is not None:
            running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
            print(f"  [PKL] Found {len(running_trials)} running trials in Optuna study.")
            for t in running_trials:
                print(f"    Patching Trial #{t.number} (ID: {t._trial_id}) state -> FAIL")
                try:
                    study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
                except AttributeError:
                    study._storage.set_trial_state(t._trial_id, optuna.trial.TrialState.FAIL)
                    
        # 2. Patch completed trials set
        completed_set = state_dict.get("_completed_trials", set())
        ot_trials_dict = state_dict.get("_ot_trials", {})
        active_ids = set(ot_trials_dict.keys()) - completed_set
        if active_ids:
            print(f"  [PKL] Adding active trial IDs to _completed_trials: {list(active_ids)}")
            completed_set.update(active_ids)
            state_dict["_completed_trials"] = completed_set
            
        # Save back
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)
        print(f"  [PKL SUCCESS] Patched: {path}")
    except Exception as e:
        print(f"  [PKL ERROR] Failed to patch {path}: {e}")

def patch_search_gen_json(path):
    print(f"\n[JSON] Processing search gen state file: {path}...")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        # 1. Patch limiter state
        limiter_state = data.get("name:ConcurrencyLimiter")
        if limiter_state is not None:
            print(f"  [JSON] Current live_trials: {limiter_state.get('live_trials')} | unfinished: {limiter_state.get('num_unfinished_live_trials')}")
            # FIX: Use actual python empty set object instead of string "set()"
            limiter_state["live_trials"] = set()
            limiter_state["num_unfinished_live_trials"] = 0
            
        # 2. Patch Experiment object to bypass PicklingErrors
        exp = data.get("experiment")
        if exp is not None and hasattr(exp, "spec") and isinstance(exp.spec, dict):
            print("  [JSON] Nullifying 'trial_dirname_creator' in Experiment spec...")
            exp.spec["trial_dirname_creator"] = None
            
        # Safe save (atomic write)
        temp_file = path + ".tmp"
        with open(temp_file, "wb") as f:
            pickle.dump(data, f)
        shutil.move(temp_file, path)
        print(f"  [JSON SUCCESS] Patched: {path}")
    except Exception as e:
        print(f"  [JSON ERROR] Failed to patch {path}: {e}")

def patch_experiment_state_json(path):
    print(f"\n[STATE] Processing experiment state JSON: {path}...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            state_data = json.load(f)
            
        modified = False
        
        # 1. Patch running trials in trial_data
        trial_data_str = state_data.get("trial_data")
        if trial_data_str:
            trial_data = json.loads(trial_data_str)
            for trial in trial_data:
                if trial.get("status") in ("RUNNING", "PENDING"):
                    print(f"  [STATE] Changing trial {trial.get('trial_id')} status from {trial.get('status')} to TERMINATED")
                    trial["status"] = "TERMINATED"
                    modified = True
            if modified:
                state_data["trial_data"] = json.dumps(trial_data)
                
        # 2. Patch runner_data if it has running list
        runner_data_str = state_data.get("runner_data")
        if runner_data_str:
            runner_data = json.loads(runner_data_str)
            # Remove any running trials from cached decisions or pending queues
            for key in ["_cached_trial_decisions", "_queued_trial_decisions"]:
                if key in runner_data and runner_data[key]:
                    print(f"  [STATE] Clearing runner_data key: {key}")
                    runner_data[key] = {}
                    modified = True
            if modified:
                state_data["runner_data"] = json.dumps(runner_data)
                
        if modified:
            # Safe save (atomic write)
            temp_file = path + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=4)
            shutil.move(temp_file, path)
            print(f"  [STATE SUCCESS] Patched and saved: {path}")
        else:
            print("  [STATE] No active trials found in experiment state. No patch needed.")
    except Exception as e:
        print(f"  [STATE ERROR] Failed to patch {path}: {e}")

def main():
    print("=== SWEEPING AND PATCHING ALL TUNE STATE LOCATIONS ===")
    
    # 1. Sweep local workspace paths - SCAN ALL MATCHING DIRECTORIES
    tune_dir = os.path.join(PROJECT_ROOT, "artifacts", "tune")
    ppo_lstm_runs = sorted(glob.glob(os.path.join(tune_dir, "tune_ppo_lstm_phase1_2026-07-*")), reverse=True)
    
    local_pkls = []
    local_jsons = []
    local_states = []
    
    # Loop over all matching run folders to patch local states
    for run_dir in ppo_lstm_runs:
        print(f"Adding local run directory to sweep: {os.path.basename(run_dir)}")
        opt_study_dir = os.path.join(run_dir, "tune_results", "optuna_study")
        local_pkls.extend(glob.glob(os.path.join(opt_study_dir, "searcher-state-*.pkl")))
        local_jsons.extend(glob.glob(os.path.join(opt_study_dir, "search_gen_state-*.json")))
        local_states.extend(glob.glob(os.path.join(opt_study_dir, "experiment_state-*.json")))
        
    # 2. Sweep /tmp ray session paths
    tmp_pkls = glob.glob("/tmp/ray/session_*/artifacts/*/optuna_study/driver_artifacts/searcher-state-*.pkl")
    tmp_jsons = glob.glob("/tmp/ray/session_*/artifacts/*/optuna_study/driver_artifacts/search_gen_state-*.json")
    tmp_states = glob.glob("/tmp/ray/session_*/artifacts/*/optuna_study/driver_artifacts/experiment_state-*.json")
    
    all_pkls = list(set(local_pkls + tmp_pkls))
    all_jsons = list(set(local_jsons + tmp_jsons))
    all_states = list(set(local_states + tmp_states))
    
    print(f"\nFound {len(all_pkls)} searcher state PKL files to patch.")
    print(f"Found {len(all_jsons)} search generator JSON files to patch.")
    print(f"Found {len(all_states)} experiment state JSON files to patch.")
    
    for pkl in all_pkls:
        patch_searcher_pkl(pkl)
        
    for js in all_jsons:
        patch_search_gen_json(js)
        
    for st in all_states:
        patch_experiment_state_json(st)
        
    print("\n[COMPLETE] All locations swept and patched successfully!")

if __name__ == "__main__":
    main()
