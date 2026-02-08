
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess
import json
import threading
import time
import re
import argparse
from datetime import datetime
from confs.model_config import GlobalConfig
from confs.env_config import EnvConfig

# ANSI Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ParallelTrainer:
    def __init__(self, run_dir, stage="TRAINING", debug=False, update_interval=None):
        self.run_dir = run_dir
        self.processes = {}
        self.progress = {}
        self.status = {}
        self.iterations = {}  # Current iteration
        self.total_iterations = {}  # Total iterations
        self.stage = stage  # TRAINING or EVALUATION
        self.default_total = GlobalConfig.TRAIN_ITERATIONS
        self.debug = debug  # Debug mode for verbose output
        self.error_messages = {}  # Store error messages
        # Update interval: use provided value, or default based on debug mode
        self.update_interval = update_interval if update_interval is not None else (10 if debug else 3)
        
    def start_training(self, name, command, output_dir):
        """Start a training process in background"""
        print(f"[{name}] Starting {self.stage.lower()}...")
        if self.debug:
            print(f"[{name}] Command: {command}")
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        self.processes[name] = process
        self.progress[name] = 0
        self.status[name] = "RUNNING"
        self.iterations[name] = 0
        self.total_iterations[name] = self.default_total
        self.error_messages[name] = []
        
        # Start thread to monitor output
        thread = threading.Thread(target=self._monitor_output, args=(name, process))
        thread.daemon = True
        thread.start()
        
        # Start thread to monitor stderr
        stderr_thread = threading.Thread(target=self._monitor_stderr, args=(name, process))
        stderr_thread.daemon = True
        stderr_thread.start()
        
    def _monitor_stderr(self, name, process):
        """Monitor stderr for error messages"""
        try:
            for line in process.stderr:
                self.error_messages[name].append(line.strip())
                if self.debug:
                    print(f"[{name} ERROR] {line.strip()}")
        except Exception:
            pass
    
    def _monitor_output(self, name, process):
        """Monitor process output and extract progress"""
        try:
            for line in process.stdout:
                # Extract episode/iteration progress
                if name == "Baseline":
                    # Look for "Episode X/Y"
                    match = re.search(r'Episode (\d+)/(\d+)', line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        self.iterations[name] = current
                        self.total_iterations[name] = total
                        self.progress[name] = int((current / total) * 100)
                else:
                    # PPO/DQN: Look for iteration count
                    match = re.search(r'iteration (\d+)', line, re.IGNORECASE)
                    if match:
                        current = int(match.group(1))
                        self.iterations[name] = current
                        self.total_iterations[name] = self.default_total
                        self.progress[name] = int((current / self.default_total) * 100)
                
                if self.debug:
                    print(f"[{name}] {line.strip()}")
            
            # Process finished
            process.wait()
            if process.returncode == 0:
                self.status[name] = "COMPLETED"
                self.progress[name] = 100
                self.iterations[name] = self.total_iterations[name]
            else:
                self.status[name] = "FAILED"
                
        except Exception as e:
            self.status[name] = "FAILED"
            self.error_messages[name].append(str(e))
            if self.debug:
                print(f"\n[{name}] Exception: {e}")
    
    def display_progress(self):
        """Display live progress in terminal"""
        while any(s == "RUNNING" for s in self.status.values()):
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"{Colors.HEADER}=" * 80)
            print(f"PARALLEL {self.stage} PROGRESS")
            print("=" * 80 + f"{Colors.ENDC}")
            print()
            
            for name in ["Baseline", "PPO", "DQN"]:
                status = self.status.get(name, "PENDING")
                progress = self.progress.get(name, 0)
                current_iter = self.iterations.get(name, 0)
                total_iter = self.total_iterations.get(name, 0)
                
                # Status icon and color
                if status == "COMPLETED":
                    icon = "[OK]"
                    color = Colors.OKGREEN
                elif status == "FAILED":
                    icon = "[XX]"
                    color = Colors.FAIL
                elif status == "RUNNING":
                    icon = "[>>]"
                    color = Colors.WARNING  # Yellow for running
                else:
                    icon = "[  ]"
                    color = Colors.ENDC
                
                # Progress bar
                bar_length = 30
                filled = int(bar_length * progress / 100)
                bar = "#" * filled + "." * (bar_length - filled)
                
                # Format iteration counter
                iter_str = f"({current_iter}/{total_iter})" if total_iter > 0 else ""
                
                # Print colored line
                print(f"{color}{icon} {name:12s} [{bar}] {progress:3d}% {iter_str:12s} {self.stage}{Colors.ENDC}")
            
            print()
            print(f"{Colors.HEADER}=" * 80 + f"{Colors.ENDC}")
            time.sleep(self.update_interval)
        
        # Final display
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.OKGREEN}=" * 80)
        print(f"{self.stage} COMPLETE")
        print("=" * 80 + f"{Colors.ENDC}")
        print()
        
        for name in ["Baseline", "PPO", "DQN"]:
            status = self.status[name]
            current_iter = self.iterations.get(name, 0)
            total_iter = self.total_iterations.get(name, 0)
            
            if status == "COMPLETED":
                print(f"{Colors.OKGREEN}[OK] {name:12s} - COMPLETED ({current_iter}/{total_iter}){Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}[FAILED] {name:12s} - FAILED{Colors.ENDC}")
                # Show last few error lines
                if name in self.error_messages and self.error_messages[name]:
                    print(f"         {Colors.FAIL}Last error: {self.error_messages[name][-1][:70]}{Colors.ENDC}")
        
        print()
        print(f"{Colors.OKGREEN}=" * 80 + f"{Colors.ENDC}")
    
    def wait_for_completion(self):
        """Wait for all processes to finish"""
        for name, process in self.processes.items():
            process.wait()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run parallel training experiments")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode with verbose output (slower updates)")
    parser.add_argument("--ui", type=int, default=None,
                       help="Terminal update interval in seconds (default: 3 for normal, 10 for debug)")
    args = parser.parse_args()
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("artifacts", timestamp)
    
    # Create directory structure
    os.makedirs(os.path.join(run_dir, "baseline", "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "ppo", "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "dqn", "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "comparison"), exist_ok=True)
    
    print("="*80)
    print(f"Starting Experiment Run: {timestamp}")
    print(f"Artifacts Directory: {run_dir}")
    if args.debug:
        print("Debug Mode: ENABLED (verbose output)")
    else:
        print("Debug Mode: DISABLED (clean output)")
    
    # Determine actual update interval
    actual_interval = args.ui if args.ui is not None else (10 if args.debug else 3)
    print(f"Update Interval: {actual_interval}s")
    print("="*80 + "\n")
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "train_iterations": GlobalConfig.TRAIN_ITERATIONS,
        "train_batch_size": GlobalConfig.TRAIN_BATCH_SIZE,
        "max_steps_per_episode": EnvConfig.MAX_STEPS,
        "episodes_per_iteration": GlobalConfig.TRAIN_BATCH_SIZE // EnvConfig.MAX_STEPS,
        "total_episodes": (GlobalConfig.TRAIN_ITERATIONS * GlobalConfig.TRAIN_BATCH_SIZE) // EnvConfig.MAX_STEPS,
        "total_steps_per_algo": GlobalConfig.TRAIN_ITERATIONS * GlobalConfig.TRAIN_BATCH_SIZE
    }
    with open(os.path.join(run_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Initialize parallel trainer with debug mode from argument
    trainer = ParallelTrainer(run_dir, stage="TRAINING", debug=args.debug, update_interval=args.ui)
    
    # Define training commands
    baseline_dir = os.path.join(run_dir, "baseline")
    ppo_dir = os.path.join(run_dir, "ppo")
    dqn_dir = os.path.join(run_dir, "dqn")
    
    # Start all training processes in parallel
    trainer.start_training(
        "Baseline",
        f"python -u scripts/train_baseline.py --output-dir {baseline_dir}",
        baseline_dir
    )
    
    trainer.start_training(
        "PPO",
        f"python -u scripts/train.py --output-dir {ppo_dir}",
        ppo_dir
    )
    
    trainer.start_training(
        "DQN",
        f"python -u scripts/train_dqn.py --output-dir {dqn_dir}",
        dqn_dir
    )
    
    # Display progress
    trainer.display_progress()
    trainer.wait_for_completion()
    
    # Check if all succeeded
    if all(s == "COMPLETED" for s in trainer.status.values()):
        print("\n✓ All training completed successfully!\n")
    else:
        print("\n✗ Some training failed. Check logs for details.\n")
        failed = [n for n, s in trainer.status.items() if s == "FAILED"]
        print(f"Failed: {', '.join(failed)}\n")
        return
    
    # Run evaluations in parallel
    print("="*80)
    print("Running Evaluations...")
    print("="*80 + "\n")
    
    eval_trainer = ParallelTrainer(run_dir, stage="EVALUATION", debug=args.debug, update_interval=args.ui)
    
    eval_trainer.start_training(
        "Baseline",
        f"python -u scripts/evaluate.py --algo Baseline --dir {baseline_dir} --no-viz --output-dir {os.path.join(baseline_dir, 'evaluation')}",
        baseline_dir
    )
    
    eval_trainer.start_training(
        "PPO",
        f"python -u scripts/evaluate.py --algo PPO --dir {ppo_dir} --no-viz --output-dir {os.path.join(ppo_dir, 'evaluation')}",
        ppo_dir
    )
    
    eval_trainer.start_training(
        "DQN",
        f"python -u scripts/evaluate.py --algo DQN --dir {dqn_dir} --no-viz --output-dir {os.path.join(dqn_dir, 'evaluation')}",
        dqn_dir
    )
    
    eval_trainer.display_progress()
    eval_trainer.wait_for_completion()
    
    # Check evaluation results
    if all(s == "COMPLETED" for s in eval_trainer.status.values()):
        print("\n✓ All evaluations completed!\n")
    else:
        print("\n✗ Some evaluations failed.\n")
        return
    
    # Generate comparison
    print("="*70)
    print("Generating Comparison Plots...")
    print("="*70 + "\n")
    
    ret = subprocess.call(f"python visualization/compare.py --run-dir {run_dir}", shell=True)
    
    if ret == 0:
        print(f"\n✓ Experiment Complete! Results in: {run_dir}\n")
    else:
        print("\n✗ Comparison generation failed.\n")

if __name__ == "__main__":
    main()
