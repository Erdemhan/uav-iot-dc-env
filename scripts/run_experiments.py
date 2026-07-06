
import os
import sys
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
from confs.model_config import GlobalConfig, QJCConfig, PPOConfig, DQNConfig, PPOLSTMConfig
from confs.env_config import EnvConfig
from confs.config import UAVConfig
from core.logger import ProcessLogger
from scripts.dashboard_server import start_dashboard, stop_dashboard, DashboardState

# ANSI Colors for terminal output
# ... (keep existing codes) ...
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
        self.stdout_log = {}
        self.loggers = {}    # ProcessLogger per algorithm
        self.is_finished = False
        # Update interval: use provided value, or default based on debug mode
        self.update_interval = update_interval if update_interval is not None else (10 if debug else 3)
        self._save_status()
        
    def _save_status(self):
        """Save trainer status to status.json on disk for dashboard consumption"""
        try:
            status_file = os.path.join(self.run_dir, "status.json")
            data = {
                "stage": self.stage,
                "is_finished": self.is_finished,
                "algorithms": {}
            }
            for name in ["Baseline", "PPO", "DQN", "PPO-LSTM"]:
                data["algorithms"][name] = {
                    "status": self.status.get(name, "PENDING"),
                    "progress": self.progress.get(name, 0),
                    "current_iteration": self.iterations.get(name, 0),
                    "total_iterations": self.total_iterations.get(name, 100),
                    "error_messages": self.error_messages.get(name, []),
                    "logs": self.stdout_log.get(name, [])
                }
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def start_training(self, name, command, output_dir):
        """Start a training process in background"""
        print(f"[{name}] Starting {self.stage.lower()}...")
        if self.debug:
            print(f"[{name}] Command: {command}")
        
        # Setup process logger
        log_file = os.path.join(output_dir, "training.log")
        self.loggers[name] = ProcessLogger(log_file)
        
        import os
        process_env = os.environ.copy()
        process_env["OMP_NUM_THREADS"] = "2"
        process_env["MKL_NUM_THREADS"] = "2"
        process_env["OPENBLAS_NUM_THREADS"] = "2"
        process_env["VECLIB_MAXIMUM_THREADS"] = "2"
        process_env["NUMEXPR_NUM_THREADS"] = "2"

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=process_env
        )
        
        self.processes[name] = process
        self.progress[name] = 0
        self.status[name] = "RUNNING"
        self.iterations[name] = 0
        self.total_iterations[name] = self.default_total
        self.error_messages[name] = []
        self.stdout_log[name] = []
        self._save_status()
        
        # Start thread to monitor output
        thread = threading.Thread(target=self._monitor_output, args=(name, process))
        thread.daemon = True
        thread.start()
        
        # Start thread to monitor stderr
        stderr_thread = threading.Thread(target=self._monitor_stderr, args=(name, process))
        stderr_thread.daemon = True
        stderr_thread.start()
        
    def _monitor_stderr(self, name, process):
        """Monitor stderr for warning/error/info messages using ProcessLogger"""
        try:
            for line in process.stderr:
                if name in self.loggers:
                    prefix, clean_line = self.loggers[name].process_line(line, "stderr")
                else:
                    prefix, clean_line = "[ERROR]", line.strip()
                
                if clean_line:
                    if prefix == "[ERROR]":
                        self.error_messages[name].append(clean_line)
                    if name not in self.stdout_log:
                        self.stdout_log[name] = []
                    self.stdout_log[name].append(f"{prefix} {clean_line}")
                    if len(self.stdout_log[name]) > 50:
                        self.stdout_log[name].pop(0)
                    self._save_status()
                if self.debug:
                    print(f"[{name} STDERR] {line.strip()}")
        except Exception:
            pass
    
    def _monitor_output(self, name, process):
        """Monitor process output and extract progress using ProcessLogger"""
        try:
            for line in process.stdout:
                if name in self.loggers:
                    prefix, clean_line = self.loggers[name].process_line(line, "stdout")
                else:
                    prefix, clean_line = "[INFO]", line.strip()
                
                if clean_line:
                    if name not in self.stdout_log:
                        self.stdout_log[name] = []
                    self.stdout_log[name].append(f"{prefix} {clean_line}")
                    if len(self.stdout_log[name]) > 50:
                        self.stdout_log[name].pop(0)
                
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
                        self._save_status()
                else:
                    # PPO/DQN: Look for iteration count
                    match = re.search(r'iteration (\d+)', line, re.IGNORECASE)
                    if match:
                        current = int(match.group(1))
                        self.iterations[name] = current
                        self.total_iterations[name] = self.default_total
                        self.progress[name] = int((current / self.default_total) * 100)
                        self._save_status()
                
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
            self._save_status()
                
        except Exception as e:
            self.status[name] = "FAILED"
            self.error_messages[name].append(str(e))
            self._save_status()
            if self.debug:
                print(f"\n[{name}] Exception: {e}")
    
    def _draw_progress_screen(self):
        print(f"{Colors.HEADER}=" * 80)
        mode_str = "PARALLEL" if not hasattr(self, "is_finished") or not self.is_finished else "SEQUENTIAL"
        print(f"{mode_str} {self.stage} PROGRESS")
        print("=" * 80 + f"{Colors.ENDC}")
        print()
        
        for name in ["Baseline", "PPO", "DQN", "PPO-LSTM"]:
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

    def display_progress(self, sequential=False):
        """Display live progress in terminal"""
        if sequential:
            while not self.is_finished:
                os.system('cls' if os.name == 'nt' else 'clear')
                self._draw_progress_screen()
                time.sleep(self.update_interval)
        else:
            while any(s == "RUNNING" for s in self.status.values()):
                os.system('cls' if os.name == 'nt' else 'clear')
                self._draw_progress_screen()
                time.sleep(self.update_interval)
        
        # Final display
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.OKGREEN}=" * 80)
        print(f"{self.stage} COMPLETE")
        print("=" * 80 + f"{Colors.ENDC}")
        print()
        
        for name in ["Baseline", "PPO", "DQN", "PPO-LSTM"]:
            status = self.status.get(name, "PENDING")
            current_iter = self.iterations.get(name, 0)
            total_iter = self.total_iterations.get(name, 0)
            
            if status == "COMPLETED":
                print(f"{Colors.OKGREEN}[OK] {name:12s} - COMPLETED ({current_iter}/{total_iter}){Colors.ENDC}")
            elif status == "FAILED":
                print(f"{Colors.FAIL}[FAILED] {name:12s} - FAILED{Colors.ENDC}")
                # Show last few error lines
                if name in self.error_messages and self.error_messages[name]:
                    print(f"         {Colors.FAIL}Last error: {self.error_messages[name][-1][:70]}{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}[PENDING] {name:12s} - NOT STARTED{Colors.ENDC}")
        
        print()
        print(f"{Colors.OKGREEN}=" * 80 + f"{Colors.ENDC}")
    
    def wait_for_completion(self):
        """Wait for all processes to finish"""
        for name, process in self.processes.items():
            process.wait()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run parallel or sequential training experiments")
    parser.add_argument("--parallel", action="store_true",
                       help="Run training and evaluation in parallel (default: sequential)")
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
    os.makedirs(os.path.join(run_dir, "ppo_lstm", "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "comparison"), exist_ok=True)
    
    print("="*80)
    print(f"Starting Experiment Run: {timestamp}")
    print(f"Artifacts Directory: {run_dir}")
    if args.parallel:
        print("Execution Mode: PARALLEL")
    else:
        print("Execution Mode: SEQUENTIAL (default)")
    if args.debug:
        print("Debug Mode: ENABLED (verbose output)")
    else:
        print("Debug Mode: DISABLED (clean output)")
    
    # Determine actual update interval
    actual_interval = args.ui if args.ui is not None else (10 if args.debug else 3)
    print(f"Update Interval: {actual_interval}s")
    print("="*80 + "\n")
    
    # Save metadata
    def get_config_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('__') and not isinstance(v, (classmethod, staticmethod)) and not callable(v)}
        
    metadata = {
        "timestamp": timestamp,
        "max_steps_per_episode": EnvConfig.MAX_STEPS,
        "episodes_per_iteration": GlobalConfig.TRAIN_BATCH_SIZE // EnvConfig.MAX_STEPS,
        "total_episodes": (GlobalConfig.TRAIN_ITERATIONS * GlobalConfig.TRAIN_BATCH_SIZE) // EnvConfig.MAX_STEPS,
        "total_steps_per_algo": GlobalConfig.TRAIN_ITERATIONS * GlobalConfig.TRAIN_BATCH_SIZE,
        "global_config": get_config_dict(GlobalConfig),
        "qjc_config": get_config_dict(QJCConfig),
        "ppo_config": get_config_dict(PPOConfig),
        "dqn_config": get_config_dict(DQNConfig),
        "ppo_lstm_config": get_config_dict(PPOLSTMConfig),
        "env_config": get_config_dict(EnvConfig),
        "uav_config": get_config_dict(UAVConfig)
    }
    with open(os.path.join(run_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Initialize parallel trainer with debug mode from argument
    trainer = ParallelTrainer(run_dir, stage="TRAINING", debug=args.debug, update_interval=args.ui)
    
    # Start Real-time Dashboard
    start_dashboard(run_dir, timestamp)
    DashboardState.current_trainer = trainer
    
    # Stop any previous Ray instances to avoid port conflicts and hung processes
    print("Stopping any existing Ray processes...")
    try:
        import subprocess
        subprocess.call("ray stop --force", shell=True)
    except Exception as e:
        print(f"Warning: Could not stop existing Ray processes: {e}")
    
    # Define training commands
    baseline_dir = os.path.join(run_dir, "baseline")
    ppo_dir = os.path.join(run_dir, "ppo")
    dqn_dir = os.path.join(run_dir, "dqn")
    ppo_lstm_dir = os.path.join(run_dir, "ppo_lstm")
    
    if args.parallel:
        # Start all training processes in parallel
        trainer.start_training(
            "Baseline",
            f"{sys.executable} -u scripts/train_baseline.py --output-dir {baseline_dir}",
            baseline_dir
        )
        
        trainer.start_training(
            "PPO",
            f"{sys.executable} -u scripts/train.py --output-dir {ppo_dir}",
            ppo_dir
        )
        
        trainer.start_training(
            "DQN",
            f"{sys.executable} -u scripts/train_dqn.py --output-dir {dqn_dir}",
            dqn_dir
        )

        trainer.start_training(
            "PPO-LSTM",
            f"{sys.executable} -u scripts/train_ppo_lstm.py --output-dir {ppo_lstm_dir}",
            ppo_lstm_dir
        )
        
        # Display progress
        trainer.display_progress(sequential=False)
        trainer.wait_for_completion()
    else:
        # Start training processes sequentially
        trainer.status = {
            "Baseline": "PENDING",
            "PPO": "PENDING",
            "DQN": "PENDING",
            "PPO-LSTM": "PENDING"
        }
        trainer.is_finished = False
        
        display_thread = threading.Thread(target=trainer.display_progress, args=(True,))
        display_thread.daemon = True
        display_thread.start()
        
        # 1. Baseline
        trainer.start_training(
            "Baseline",
            f"{sys.executable} -u scripts/train_baseline.py --output-dir {baseline_dir}",
            baseline_dir
        )
        while trainer.status["Baseline"] in ["PENDING", "RUNNING"]:
            time.sleep(0.5)
        if trainer.status["Baseline"] == "FAILED":
            trainer.is_finished = True
            display_thread.join()
            
        # 2. PPO
        if trainer.status["Baseline"] == "COMPLETED":
            trainer.start_training(
                "PPO",
                f"{sys.executable} -u scripts/train.py --output-dir {ppo_dir}",
                ppo_dir
            )
            while trainer.status["PPO"] in ["PENDING", "RUNNING"]:
                time.sleep(0.5)
            if trainer.status["PPO"] == "FAILED":
                trainer.is_finished = True
                display_thread.join()
                
        # 3. DQN
        if trainer.status["PPO"] == "COMPLETED":
            trainer.start_training(
                "DQN",
                f"{sys.executable} -u scripts/train_dqn.py --output-dir {dqn_dir}",
                dqn_dir
            )
            while trainer.status["DQN"] in ["PENDING", "RUNNING"]:
                time.sleep(0.5)
            if trainer.status["DQN"] == "FAILED":
                trainer.is_finished = True
                display_thread.join()
                
        # 4. PPO-LSTM
        if trainer.status["DQN"] == "COMPLETED":
            trainer.start_training(
                "PPO-LSTM",
                f"{sys.executable} -u scripts/train_ppo_lstm.py --output-dir {ppo_lstm_dir}",
                ppo_lstm_dir
            )
            while trainer.status["PPO-LSTM"] in ["PENDING", "RUNNING"]:
                time.sleep(0.5)
                
        trainer.is_finished = True
        display_thread.join()
    
    # Check if all succeeded
    if all(s == "COMPLETED" for s in trainer.status.values()):
        print("\n[OK] All training completed successfully!\n")
    else:
        print("\n[FAIL] Some training failed. Check logs for details.\n")
        failed = [n for n, s in trainer.status.items() if s == "FAILED"]
        print(f"Failed: {', '.join(failed)}\n")
        trainer.stage = "FAILED"
        trainer.is_finished = True
        trainer._save_status()
        DashboardState.stage = "FAILED"
        DashboardState.end_time = time.time()
        return
    
    # Run robustness evaluation directly
    print("="*80)
    print("Running 30-seed Robustness Evaluation (30 random seeds)...")
    print("="*80 + "\n")
    
    trainer.stage = "ROBUSTNESS"
    trainer._save_status()
    DashboardState.stage = "ROBUSTNESS"
    DashboardState.current_trainer = None
    
    ret_robust = subprocess.call(f"{sys.executable} scripts/evaluate_paper_robustness.py --run-dir {run_dir}", shell=True)
    
    if ret_robust == 0:
        trainer.stage = "COMPLETED"
        trainer.is_finished = True
        trainer._save_status()
        DashboardState.stage = "COMPLETED"
        DashboardState.end_time = time.time()
        print(f"\n[OK] Experiment & Robustness Evaluation Complete! Results in: {run_dir}\n")
    else:
        trainer.stage = "FAILED"
        trainer.is_finished = True
        trainer._save_status()
        DashboardState.stage = "FAILED"
        DashboardState.end_time = time.time()
        print("\n[FAIL] Robustness evaluation failed.\n")
        
    print("\n[DASHBOARD] Press Enter in this console to exit and close the dashboard...")
    input()
    stop_dashboard()
    
    # Ray shutdown no longer needed since processes run independently
    pass

if __name__ == "__main__":
    main()
