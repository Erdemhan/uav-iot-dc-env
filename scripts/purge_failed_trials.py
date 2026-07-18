import pickle
import os
import glob
import sys
import shutil
import json
import optuna

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
            
        # Parse last line to get actual training_iteration value
        last_line = json.loads(lines[-1].strip())
        iter_count = last_line.get("training_iteration", 0)
        
        if iter_count >= 999:
            return True
    except Exception as e:
        print(f"      [WARN] Could not parse result.json for {os.path.basename(logdir)}: {e}")
        
    return False

def purge_run_directory(run_dir, is_temp=False):
    print(f"\n=== PURGING RUN DIRECTORY ({'TEMP' if is_temp else 'LOCAL'}): {run_dir} ===")
    
    opt_study_dir = os.path.join(run_dir, "tune_results", "optuna_study")
    if is_temp:
        opt_study_dir = os.path.join(run_dir, "optuna_study", "driver_artifacts")
        
    pkl_files = sorted(glob.glob(os.path.join(opt_study_dir, "searcher-state-*.pkl")))
    json_files = sorted(glob.glob(os.path.join(opt_study_dir, "search_gen_state-*.json")))
    state_files = sorted(glob.glob(os.path.join(opt_study_dir, "experiment_state-*.json")))
    
    if not pkl_files or not state_files:
        print("  [WARN] Required state files not found. Skipping...")
        return
        
    pkl_path = pkl_files[-1]
    json_path = json_files[-1] if json_files else None
    state_path = state_files[-1]
    
    # Backup files first before doing surgery
    print(f"  [BACKUP] Creating safety backups...")
    shutil.copy2(pkl_path, pkl_path + ".bak")
    shutil.copy2(state_path, state_path + ".bak")
    if json_path:
        shutil.copy2(json_path, json_path + ".bak")
        
    # 1. Load Experiment State JSON to inspect trials
    print(f"  [STATE] Inspecting trials in state JSON: {state_path}")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state_data = json.load(f)
            
        trial_data_raw = state_data.get("trial_data")
        kept_trials = []
        purged_trial_ids = set()
        
        if trial_data_raw is not None:
            for item in trial_data_raw:
                if isinstance(item, list) and len(item) > 0:
                    elem = item[0]
                    if isinstance(elem, str):
                        try:
                            trial_dict = json.loads(elem)
                        except:
                            continue
                    elif isinstance(elem, dict):
                        trial_dict = elem
                    else:
                        continue
                        
                    trial_id = trial_dict.get("trial_id")
                    
                    # DYNAMIC PATH RESOLUTION:
                    # Instead of using the stored 'logdir' path which might be from another machine,
                    # we resolve the path directly inside our local workspace relative to PROJECT_ROOT.
                    local_logdir = os.path.join(PROJECT_ROOT, "artifacts", "tune", "tune_ppo_lstm_phase1_2026-07-17_10-29-00", "tune_results", "optuna_study", f"trial_{trial_id}")
                    
                    is_valid = False
                    status = trial_dict.get("status")
                    if status == "TERMINATED":
                        # Check local workspace result.json
                        if verify_trial_iterations(local_logdir):
                            is_valid = True
                        else:
                            print(f"    Purging trial {trial_id} (Incomplete iterations: reached < 999)")
                    else:
                        print(f"    Purging active/failed trial {trial_id} (Status is {status})")
                        
                    if is_valid:
                        kept_trials.append(item)
                    else:
                        purged_trial_ids.add(trial_id)
                        
            state_data["trial_data"] = kept_trials
            print(f"  [STATE] Kept {len(kept_trials)} trials in JSON (purged {len(purged_trial_ids)}).")
            
        # Clean runner_data
        runner_data_raw = state_data.get("runner_data")
        if runner_data_raw is not None:
            if isinstance(runner_data_raw, str):
                runner_data = json.loads(runner_data_raw)
                runner_data_is_str = True
            else:
                runner_data = runner_data_raw
                runner_data_is_str = False
                
            for key in ["_cached_trial_decisions", "_queued_trial_decisions"]:
                if key in runner_data:
                    runner_data[key] = {}
                    
            if runner_data_is_str:
                state_data["runner_data"] = json.dumps(runner_data)
            else:
                state_data["runner_data"] = runner_data
            
        # Save JSON
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=4)
        print("  [STATE SUCCESS] State JSON successfully purged and saved.")
    except Exception as e:
        print(f"  [STATE ERROR] Failed: {e}")
        return
        
    # 2. Load and purge Optuna PKL using the kept trial IDs
    print(f"  [PKL] Purging failed trials from PKL: {pkl_path}")
    try:
        with open(pkl_path, "rb") as f:
            state_dict = pickle.load(f)
            
        old_study = state_dict.get("_ot_study")
        if old_study is None:
            print("  [PKL ERROR] Optuna study not found in state.")
            return
            
        completed_trials = []
        for t in old_study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                is_kept = False
                for item in kept_trials:
                    elem = item[0]
                    if isinstance(elem, str):
                        td = json.loads(elem)
                    else:
                        td = elem
                    if t._trial_id == td.get("trial_id") or f"trial_{t._trial_id}" in td.get("logdir", ""):
                        is_kept = True
                        break
                if is_kept:
                    completed_trials.append(t)
                    
        completed_trial_ids = {t._trial_id for t in completed_trials}
        print(f"  [PKL] Found {len(completed_trials)} truly complete (1000/1000 iters) trials in study.")
        
        new_study = optuna.create_study(study_name=old_study.study_name, direction=old_study.direction)
        for t in completed_trials:
            new_study.add_trial(t)
            
        old_ot_trials = state_dict.get("_ot_trials", {})
        new_ot_trials = {tid: ot_trial for tid, ot_trial in old_ot_trials.items() if tid in completed_trial_ids}
        
        state_dict["_ot_study"] = new_study
        state_dict["_ot_trials"] = new_ot_trials
        state_dict["_completed_trials"] = completed_trial_ids
        
        with open(pkl_path, "wb") as f:
            pickle.dump(state_dict, f)
        print("  [PKL SUCCESS] PKL successfully purged and saved.")
    except Exception as e:
        print(f"  [PKL ERROR] Failed: {e}")
        return
        
    # 3. Purge ConcurrencyLimiter JSON
    if json_path:
        print(f"  [LIMITER] Purging ConcurrencyLimiter: {json_path}")
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
            print("  [LIMITER SUCCESS] ConcurrencyLimiter successfully purged.")
        except Exception as e:
            print(f"  [LIMITER ERROR] Failed: {e}")
            
    # 4. Delete physical trial folders of purged trials from disk
    if not is_temp:
        # FIX: Physical trial folders are in opt_study_dir's parent (tune_results/optuna_study/)
        for tid in purged_trial_ids:
            trial_folders = glob.glob(os.path.join(opt_study_dir, f"trial_{tid}*"))
            for tf in trial_folders:
                print(f"  [DISK] Deleting corrupted/purged trial folder: {tf}")
                try:
                    shutil.rmtree(tf)
                except Exception as de:
                    print(f"    [DISK ERROR] Could not delete {tf}: {de}")

def main():
    print("=== STARTING SURGICAL PURGE OF INCOMPLETE & FAILED TRIALS (ITER < 999) ===")
    
    local_run_dir = os.path.join(PROJECT_ROOT, "artifacts", "tune", "tune_ppo_lstm_phase1_2026-07-17_10-29-00")
    if os.path.exists(local_run_dir):
        purge_run_directory(local_run_dir, is_temp=False)
    else:
        print(f"[ERROR] Local target run directory not found: {local_run_dir}")
        
    temp_run_dir = "/tmp/ray/session_2026-07-17_10-05-50_996511_3860/artifacts/2026-07-17_10-29-00"
    if os.path.exists(temp_run_dir):
        purge_run_directory(temp_run_dir, is_temp=True)
    else:
        print(f"[INFO] Temp run directory not found or already cleaned: {temp_run_dir}")
        
    print("\n[PURGE COMPLETE] Cerrahi temizlik basariyla tamamlandi!")
    print("Artik '--num-samples 100' ile baslatabilirsiniz. Optuna 100 adet temiz tamamlanmis trial uretecektir.")

if __name__ == "__main__":
    main()
