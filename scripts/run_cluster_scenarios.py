import subprocess
import time
import sys
import os
from datetime import datetime

def run_cmd(cmd, log_file):
    """Start process and pipe outputs to log file"""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    f = open(log_file, "w", encoding="utf-8")
    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True
    )
    return p, f

def wait_for_processes(processes_dict):
    """Wait for a set of processes to finish and report exit statuses"""
    print("\nWaiting for processes to complete...")
    active = list(processes_dict.keys())
    last_heartbeat = time.time()
    
    while active:
        for name in list(active):
            p, f, log_file = processes_dict[name]
            exit_code = p.poll()
            if exit_code is not None:
                f.close()
                active.remove(name)
                if exit_code == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Process {name} finished with status: SUCCESS")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Process {name} finished with status: FAILED (code {exit_code}) (Log: {log_file})")
                    # Print last 5 lines of log file directly to console on failure
                    try:
                        if os.path.exists(log_file):
                            with open(log_file, "r", encoding="utf-8", errors="ignore") as lf:
                                lines = [line.strip() for line in lf.readlines() if line.strip()]
                                tail_lines = lines[-5:] if len(lines) >= 5 else lines
                                print(f"--- [ERROR LOG TAIL: {name}] ---")
                                for line in tail_lines:
                                    print(f"  {line}")
                                print("-----------------------------------")
                    except Exception:
                        pass

        # Print heartbeat every 20 seconds so user knows jobs are actively training
        if active and (time.time() - last_heartbeat) >= 20:
            active_list_str = ", ".join(active)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cluster Status] Training in progress across {len(active)} active job(s): {active_list_str}")
            last_heartbeat = time.time()

        time.sleep(5)
    print("All processes in this batch have finished!\n")

def verify_checkpoints(run_dir, scenario_name):
    """Verifies that checkpoint folders exist and reports their sizes on disk"""
    print("\n" + "=" * 80)
    print(f"  CHECKPOINT VERIFICATION SUMMARY ({scenario_name})")
    print("=" * 80)
    
    algos = ["ppo", "dqn", "ppo_lstm"]
    for algo in algos:
        algo_dir = os.path.join(run_dir, algo)
        ckpt_dir = os.path.join(algo_dir, "checkpoint_001000")
        
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(ckpt_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            size_mb = total_size / (1024 * 1024)
            print(f"  [OK] {scenario_name} | {algo.upper():<10} -> Checkpoint VERIFIED! Location: {ckpt_dir} ({size_mb:.2f} MB)")
        else:
            print(f"  [WARNING] {scenario_name} | {algo.upper():<10} -> Checkpoint MISSING at: {ckpt_dir}")
    print("=" * 80 + "\n")

def copy_ray_progress_to_run_dir(scenario, algo, target_dir):
    """Finds the latest progress.csv under ~/ray_results/{algo_prefix}_{scenario}/t_*/progress.csv and copies it"""
    import shutil
    import glob
    home = os.path.expanduser("~")
    ray_results_dir = os.path.join(home, "ray_results")
    
    algo_prefixes = {
        "ppo": "PPO",
        "dqn": "DQN",
        "ppo_lstm": "PPO_LSTM"
    }
    prefix = algo_prefixes.get(algo)
    if not prefix:
        return
        
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Copy progress.csv
    search_pattern = os.path.join(ray_results_dir, f"{prefix}_{scenario}", "t_*", "progress.csv")
    csv_files = glob.glob(search_pattern)
    if csv_files:
        newest_csv = max(csv_files, key=os.path.getmtime)
        shutil.copy2(newest_csv, os.path.join(target_dir, "progress.csv"))
        print(f"Successfully copied {newest_csv} -> {os.path.join(target_dir, 'progress.csv')}")
    else:
        print(f"Warning: No progress.csv found for {scenario} {algo} using pattern: {search_pattern}")
        
    # 2. Copy latest checkpoint folder
    search_ckpt = os.path.join(ray_results_dir, f"{prefix}_{scenario}", "t_*", "checkpoint_*")
    ckpt_dirs = [d for d in glob.glob(search_ckpt) if os.path.isdir(d)]
    if ckpt_dirs:
        newest_ckpt = max(ckpt_dirs, key=os.path.getmtime)
        dest_ckpt = os.path.join(target_dir, os.path.basename(newest_ckpt))
        if os.path.exists(dest_ckpt):
            shutil.rmtree(dest_ckpt)
        shutil.copytree(newest_ckpt, dest_ckpt)
        print(f"Successfully copied checkpoint {newest_ckpt} -> {dest_ckpt}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Orchestrate Scenario 1 and 2 runs across the 6-PC Ray Cluster")
    parser.add_argument("--ray-address", type=str, default="auto", 
                        help="Ray cluster address. Default 'auto' to run all 6 jobs concurrently across the 6-PC cluster.")
    parser.add_argument("--skip-qjc", action="store_true", help="Skip running local QJC baseline")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("artifacts", "scenario_runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Auto-start Web Dashboard pointing to active run_dir
    try:
        from scripts.dashboard_server import start_dashboard
        start_dashboard(run_dir, timestamp, page="index.html")
        print(f"[DASHBOARD] Web Dashboard server automatically launched at http://localhost:5000 (Run Dir: {run_dir})")
    except Exception as e:
        print(f"[DASHBOARD] Warning: Could not auto-start dashboard: {e}")
    
    ray_arg = f"--ray-address {args.ray_address}" if args.ray_address else ""

    print("=" * 80)
    print(f"Ray Cluster Orchestrator: {timestamp}")
    print(f"Ray Address: {args.ray_address}")
    print(f"Logs Directory: {run_dir}")
    print("=" * 80)

    # ----------------------------------------------------
    # PHASE 1: SCENARIO 1 (15 Nodes, 1 UAV, 500m)
    # ----------------------------------------------------
    print("\n" + "#"*80)
    print("  PHASE 1: RUNNING SCENARIO 1 (1-A and 1-B)")
    print("  Scheduling jobs (PPO, DQN, PPO-LSTM x A/B)")
    print("#"*80 + "\n")
    
    def make_job_cmd(algo, scenario, output_dir):
        ray_arg = f"--ray-address {args.ray_address}" if args.ray_address else ""
        return f"{sys.executable} -u scripts/train.py --algo {algo} --scenario {scenario} {ray_arg} --output-dir {output_dir}".strip()

    s1_jobs = {
        # Scenario 1-A (Low Power Penalty: w_cost = 0.03)
        "S1-A_PPO": (
            make_job_cmd("ppo", "1-A", os.path.join(run_dir, 'S1-A', 'ppo')),
            os.path.join(run_dir, "S1-A", "ppo.log")
        ),
        "S1-A_DQN": (
            make_job_cmd("dqn", "1-A", os.path.join(run_dir, 'S1-A', 'dqn')),
            os.path.join(run_dir, "S1-A", "dqn.log")
        ),
        "S1-A_PPO-LSTM": (
            make_job_cmd("ppo_lstm", "1-A", os.path.join(run_dir, 'S1-A', 'ppo_lstm')),
            os.path.join(run_dir, "S1-A", "ppo_lstm.log")
        ),
        # Scenario 1-B (High Power Penalty: w_cost = 0.3)
        "S1-B_PPO": (
            make_job_cmd("ppo", "1-B", os.path.join(run_dir, 'S1-B', 'ppo')),
            os.path.join(run_dir, "S1-B", "ppo.log")
        ),
        "S1-B_DQN": (
            make_job_cmd("dqn", "1-B", os.path.join(run_dir, 'S1-B', 'dqn')),
            os.path.join(run_dir, "S1-B", "dqn.log")
        ),
        "S1-B_PPO-LSTM": (
            make_job_cmd("ppo_lstm", "1-B", os.path.join(run_dir, 'S1-B', 'ppo_lstm')),
            os.path.join(run_dir, "S1-B", "ppo_lstm.log")
        )
    }

    active_s1 = {}
    for name, (cmd, log_file) in s1_jobs.items():
        print(f"Launching {name}...")
        p, f = run_cmd(cmd, log_file)
        active_s1[name] = (p, f, log_file)
        
    wait_for_processes(active_s1)
    
    # Verify Scenario 1 checkpoints on disk
    for scen in ["1-A", "1-B"]:
        verify_checkpoints(os.path.join(run_dir, f"S{scen}"), f"Scenario {scen}")

    # ----------------------------------------------------
    # LOCAL BASELINES FOR SCENARIO 1 (QJC)
    # ----------------------------------------------------
    if not args.skip_qjc:
        print("\n" + "-"*80)
        print("  Running QJC Local Baselines for Scenario 1")
        print("-"*80 + "\n")
        qjc_jobs = {
            "S1-A_QJC": f"{sys.executable} -u scripts/train_baseline.py --scenario 1-A --output-dir {os.path.join(run_dir, 'S1-A', 'qjc')}",
            "S1-B_QJC": f"{sys.executable} -u scripts/train_baseline.py --scenario 1-B --output-dir {os.path.join(run_dir, 'S1-B', 'qjc')}"
        }
        for name, cmd in qjc_jobs.items():
            print(f"Running {name} locally...")
            subprocess.run(cmd, shell=True)
            print(f"{name} completed.")

    # ----------------------------------------------------
    # PHASE 2: SCENARIO 2 (30 Nodes, 2 UAVs, 1000m)
    # ----------------------------------------------------
    print("\n" + "#"*80)
    print("  PHASE 2: RUNNING SCENARIO 2 (2-A and 2-B)")
    print("  Scheduling 6 concurrent jobs to 6 Worker PCs (PPO, DQN, PPO-LSTM x A/B)")
    print("#"*80 + "\n")
    
    s2_jobs = {
        # Scenario 2-A (Low Power Penalty: w_cost = 0.03)
        "S2-A_PPO": (
            make_job_cmd("ppo", "2-A", os.path.join(run_dir, 'S2-A', 'ppo')),
            os.path.join(run_dir, "S2-A", "ppo.log")
        ),
        "S2-A_DQN": (
            make_job_cmd("dqn", "2-A", os.path.join(run_dir, 'S2-A', 'dqn')),
            os.path.join(run_dir, "S2-A", "dqn.log")
        ),
        "S2-A_PPO-LSTM": (
            make_job_cmd("ppo_lstm", "2-A", os.path.join(run_dir, 'S2-A', 'ppo_lstm')),
            os.path.join(run_dir, "S2-A", "ppo_lstm.log")
        ),
        # Scenario 2-B (High Power Penalty: w_cost = 0.3)
        "S2-B_PPO": (
            make_job_cmd("ppo", "2-B", os.path.join(run_dir, 'S2-B', 'ppo')),
            os.path.join(run_dir, "S2-B", "ppo.log")
        ),
        "S2-B_DQN": (
            make_job_cmd("dqn", "2-B", os.path.join(run_dir, 'S2-B', 'dqn')),
            os.path.join(run_dir, "S2-B", "dqn.log")
        ),
        "S2-B_PPO-LSTM": (
            make_job_cmd("ppo_lstm", "2-B", os.path.join(run_dir, 'S2-B', 'ppo_lstm')),
            os.path.join(run_dir, "S2-B", "ppo_lstm.log")
        )
    }

    active_s2 = {}
    for name, (cmd, log_file) in s2_jobs.items():
        print(f"Launching {name}...")
        p, f = run_cmd(cmd, log_file)
        active_s2[name] = (p, f, log_file)
        
    wait_for_processes(active_s2)
    
    # Verify Scenario 2 checkpoints on disk
    for scen in ["2-A", "2-B"]:
        verify_checkpoints(os.path.join(run_dir, f"S{scen}"), f"Scenario {scen}")

    # ----------------------------------------------------
    # LOCAL BASELINES FOR SCENARIO 2 (QJC)
    # ----------------------------------------------------
    if not args.skip_qjc:
        print("\n" + "-"*80)
        print("  Running QJC Local Baselines for Scenario 2")
        print("-"*80 + "\n")
        qjc_jobs = {
            "S2-A_QJC": f"{sys.executable} -u scripts/train_baseline.py --scenario 2-A --output-dir {os.path.join(run_dir, 'S2-A', 'qjc')}",
            "S2-B_QJC": f"{sys.executable} -u scripts/train_baseline.py --scenario 2-B --output-dir {os.path.join(run_dir, 'S2-B', 'qjc')}"
        }
        for name, cmd in qjc_jobs.items():
            print(f"Running {name} locally...")
            subprocess.run(cmd, shell=True)
            print(f"{name} completed.")

    # ----------------------------------------------------
    # PHASE 3: GENERATING REWARD LEARNING CURVES
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("  PHASE 3: GENERATING REWARD LEARNING CURVES")
    print("="*80 + "\n")
    
    plot_cmd = f"{sys.executable} scripts/plot_scenario_learning_curves.py --run-dir {run_dir}"
    subprocess.run(plot_cmd, shell=True)

    # ----------------------------------------------------
    # PHASE 4: AUTOMATIC ROBUSTNESS EVALUATION & PLOTS
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("  PHASE 4: AUTOMATIC ROBUSTNESS EVALUATION & PLOT GENERATION")
    print("="*80 + "\n")
    
    scenario_configs = {
        "1-A": {"NUM_NODES": 15, "NUM_UAVS": 1, "AREA_SIZE": 500.0, "W_COST": 0.03},
        "1-B": {"NUM_NODES": 15, "NUM_UAVS": 1, "AREA_SIZE": 500.0, "W_COST": 0.3},
        "2-A": {"NUM_NODES": 30, "NUM_UAVS": 2, "AREA_SIZE": 1000.0, "W_COST": 0.03},
        "2-B": {"NUM_NODES": 30, "NUM_UAVS": 2, "AREA_SIZE": 1000.0, "W_COST": 0.3},
    }
    
    import json
    for scen in ["1-A", "1-B", "2-A", "2-B"]:
        scen_dir = os.path.join(run_dir, f"S{scen}")
        if os.path.exists(scen_dir):
            meta_path = os.path.join(scen_dir, "metadata.json")
            if not os.path.exists(meta_path):
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({"scenario": scen, "env_config": scenario_configs[scen]}, f, indent=2)
            
            print(f"\n---> Evaluating Robustness & Generating Plots for Scenario {scen}...")
            eval_cmd = f"{sys.executable} scripts/evaluate_paper_robustness.py --run-dir {scen_dir}"
            subprocess.run(eval_cmd, shell=True)

    print("\n" + "="*80)
    print("  ALL SCENARIOS & EVALUATIONS COMPLETED SUCCESSFULLY!")
    print(f"  Artifacts, Robustness Evaluation, and Plots saved under: {run_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
