
import os
import subprocess
import shutil
import glob
import time

def get_latest_log_dir(logs_root="logs"):
    all_dirs = glob.glob(os.path.join(logs_root, "EXP_*"))
    if not all_dirs:
        return None
    return max(all_dirs, key=os.path.getmtime)

def run_step(command, name, output_folder):
    print(f"--- Running {name} ---")
    print(f"Command: {command}")
    
    # 1. Capture start time to find new log
    start_time = time.time()
    
    # 2. Run Command
    # We use subprocess.call to wait for completion
    ret = subprocess.call(command, shell=True)
    
    if ret != 0:
        print(f"Error: {name} failed with code {ret}")
        return False
        
    # 3. Find produced log
    # We look for the folder modified/created *after* start_time
    # Or just simply get the absolute latest folder in logs/
    latest_log = get_latest_log_dir()
    
    if not latest_log:
        print(f"Warning: No log generated for {name}")
        return False
        
    print(f"Generated Log: {latest_log}")
    
    # 4. Move/Rename to experiment folder
    target_dir = os.path.join("experiments", output_folder)
    
    # Remove existing if any
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    # Copy or Move? Move is better to keep logs clean
    shutil.move(latest_log, target_dir)
    print(f"Saved Results to: {target_dir}")
    print("\n")
    return True

def main():
    # Ensure experiments dir exists
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
        
    # 1. Baseline (QJC)
    # Train
    print("--- Training Baseline (QJC) ---")
    subprocess.call("python train_baseline.py", shell=True)
    # Evaluate (Main.py now loads the trained model)
    # Run HEADLESS (No Viz)
    run_step("python main.py --no-viz", "Baseline Evaluation", "baseline")
    
    # 2. PPO (Train + Eval)
    # Train
    print("--- Training PPO ---")
    subprocess.call("python train.py", shell=True) 
    # Evaluate (Produces log)
    run_step("python evaluate.py --algo PPO --dir ./ray_results --no-viz", "PPO Evaluation", "ppo")
    
    # 3. DQN (Train + Eval)
    # Train
    print("--- Training DQN ---")
    subprocess.call("python train_dqn.py", shell=True) 
    # Evaluate (Produces log)
    run_step("python evaluate.py --algo DQN --dir ./ray_results_dqn --no-viz", "DQN Evaluation", "dqn")
    
    print("\n" + "="*60)
    print("All Experiments Completed. Generating Comparison...")
    print("="*60 + "\n")
    
    # Automatically run comparison visualization
    subprocess.call("python visualization/compare.py", shell=True)

if __name__ == "__main__":
    main()
