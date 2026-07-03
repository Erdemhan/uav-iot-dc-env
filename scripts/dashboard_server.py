import os
import json
import http.server
import socketserver
import threading
import webbrowser
import time
import glob
import csv

class DashboardState:
    run_dir = ""
    stage = "TRAINING"
    current_trainer = None
    timestamp = ""
    port = 5000
    server_thread = None
    server_instance = None
    start_time = 0.0
    end_time = None

def parse_rllib_progress(csv_path):
    history = []
    if not csv_path or not os.path.exists(csv_path):
        return history, None, None
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                iteration = int(row.get('training_iteration', 0))
                steps = int(row.get('timesteps_total', 0))
                reward = row.get('env_runners/episode_reward_mean') or row.get('episode_reward_mean')
                if reward is not None and reward != '':
                    reward = float(reward)
                else:
                    reward = 0.0
                history.append({
                    'iteration': iteration,
                    'steps': steps,
                    'reward': reward
                })
        if history:
            latest = history[-1]
            return history, latest['reward'], latest['steps']
    except Exception as e:
        # Silently ignore read conflicts during training writes
        pass
    return history, None, None

def parse_baseline_progress(csv_path):
    history = []
    if not csv_path or not os.path.exists(csv_path):
        return history, None, None
    try:
        raw_episodes = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep = int(row.get('episode', 0))
                tot_r = float(row.get('total_reward', 0.0))
                raw_episodes.append((ep, tot_r))
        
        # Resample every 10 episodes (which equals 1000 steps) to match Deep RL iteration batch size
        bin_size = 10
        steps_per_episode = 100
        for i in range(0, len(raw_episodes), bin_size):
            chunk = raw_episodes[i : i + bin_size]
            if not chunk:
                continue
            avg_r = sum(item[1] for item in chunk) / len(chunk)
            last_ep = chunk[-1][0]
            steps = last_ep * steps_per_episode
            iteration = (i // bin_size) + 1
            history.append({
                'iteration': iteration,
                'steps': steps,
                'reward': avg_r
            })
        if history:
            latest = history[-1]
            return history, latest['reward'], latest['steps']
    except Exception as e:
        pass
    return history, None, None
def get_active_run_dir():
    """Dynamically get the active run directory. Supports both training runs and Optuna tuning runs."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    active_file = os.path.join(project_root, "dashboard_active_run.txt")
    if os.path.exists(active_file):
        try:
            with open(active_file, "r", encoding="utf-8") as f:
                path = f.read().strip()
                if os.path.exists(path):
                    return path
        except:
            pass
    return DashboardState.run_dir

class DashboardHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    # Overwrite log_message to keep stdout clean
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = self.path.split('?')[0]
        run_dir = get_active_run_dir()
        
        # Serve main index.html
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            # Locate index.html next to this file in scripts/dashboard/
            base_dir = os.path.dirname(os.path.abspath(__file__))
            html_path = os.path.join(base_dir, "dashboard", "index.html")
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(b"<h1>Dashboard index.html not found!</h1>")
            return

        # Serve metadata
        if path == "/api/metadata":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            metadata_path = os.path.join(run_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(json.dumps({}).encode("utf-8"))
            return

        # Serve robustness results
        if path == "/api/robustness":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            
            robustness_path = os.path.join(run_dir, "comparison", "robustness_results_30seeds.json")
            if os.path.exists(robustness_path):
                with open(robustness_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(json.dumps({}).encode("utf-8"))
            return

        # Serve robustness.html
        if path == "/robustness.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            html_path = os.path.join(base_dir, "dashboard", "robustness.html")
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(b"<h1>robustness.html not found!</h1>")
            return

        # Serve robustness comparison plot
        if path == "/api/plots/robustness.png":
            img_path = os.path.join(run_dir, "comparison", "comparison_robustness.png")
            if os.path.exists(img_path):
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            return

        # Serve loss decomposition plot
        if path == "/api/plots/loss_decomposition.png":
            img_path = os.path.join(run_dir, "comparison", "loss_decomposition.png")
            if os.path.exists(img_path):
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            return

        # Serve opt.html
        if path == "/opt.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            html_path = os.path.join(base_dir, "dashboard", "opt.html")
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(b"<h1>opt.html not found!</h1>")
            return

        # Serve active Optuna trial progress (Ray Tune progress.csv per trial)
        if path == "/api/active_trials":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()

            active_trials = []
            tune_results_dir = os.path.join(run_dir, "tune_results", "optuna_study")
            if os.path.isdir(tune_results_dir):
                for trial_folder in sorted(os.listdir(tune_results_dir)):
                    trial_path = os.path.join(tune_results_dir, trial_folder)
                    if not os.path.isdir(trial_path):
                        continue
                    progress_csv = os.path.join(trial_path, "progress.csv")
                    if not os.path.exists(progress_csv):
                        continue
                    try:
                        with open(progress_csv, "r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                        if not rows:
                            continue
                        last = rows[-1]
                        current_iter = int(float(last.get("training_iteration", 0) or 0))
                        objective = last.get("objective", None)
                        jsr = last.get("jsr", None)
                        if objective not in (None, ""):
                            objective = float(objective)
                        else:
                            objective = None
                        if jsr not in (None, ""):
                            jsr = float(jsr)
                        else:
                            jsr = None
                        # Read total_iterations from params json if exists
                        params_path = os.path.join(trial_path, "params.json")
                        total_iters = None
                        if os.path.exists(params_path):
                            try:
                                with open(params_path, "r", encoding="utf-8") as pf:
                                    p = json.load(pf)
                                    total_iters = p.get("iterations")
                            except:
                                pass
                        active_trials.append({
                            "trial_id": trial_folder,
                            "current_iteration": current_iter,
                            "total_iterations": total_iters,
                            "objective": objective,
                            "jsr": jsr,
                            "num_rows": len(rows)
                        })
                    except Exception:
                        pass

            self.wfile.write(json.dumps({"trials": active_trials}).encode("utf-8"))
            return

        # Serve Optuna results JSON
        if path == "/api/optuna":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            
            trials_path = os.path.join(run_dir, "optuna", "optuna_trials.json")
            best_path = os.path.join(run_dir, "optuna", "best_params.json")
            
            response_data = {"trials": [], "best_value": None, "best_trial_number": None}
            if os.path.exists(trials_path):
                try:
                    with open(trials_path, "r", encoding="utf-8") as f:
                        response_data["trials"] = json.load(f)
                except:
                    pass
            if os.path.exists(best_path):
                try:
                    with open(best_path, "r", encoding="utf-8") as f:
                        best = json.load(f)
                        response_data["best_value"] = best.get("best_value")
                        response_data["best_trial_number"] = best.get("best_trial_number")
                except:
                    pass
                    
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        # Serve Optuna plots
        if path.startswith("/api/plots/optuna/"):
            plot_name = os.path.basename(path)
            img_path = os.path.join(run_dir, "optuna", plot_name)
            if os.path.exists(img_path):
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            return

        # Serve real-time progress
        if path == "/api/progress":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            # Avoid caching
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            
            # Calculate elapsed time
            elapsed = 0.0
            if DashboardState.start_time > 0:
                if DashboardState.end_time:
                    elapsed = DashboardState.end_time - DashboardState.start_time
                else:
                    elapsed = time.time() - DashboardState.start_time
            
            # Gather progress data
            response_data = {
                "timestamp": DashboardState.timestamp,
                "stage": DashboardState.stage,
                "elapsed_time": elapsed,
                "algorithms": {}
            }
            
            trainer = DashboardState.current_trainer
            
            # Map algorithm names to folders and config keys
            algos = {
                "Baseline": {
                    "csv_finder": lambda: os.path.join(run_dir, "baseline", "training_curve.csv"),
                    "parser": parse_baseline_progress
                },
                "PPO": {
                    "csv_finder": lambda: self._find_rllib_progress(run_dir, "ppo"),
                    "parser": parse_rllib_progress
                },
                "DQN": {
                    "csv_finder": lambda: self._find_rllib_progress(run_dir, "dqn"),
                    "parser": parse_rllib_progress
                },
                "PPO-LSTM": {
                    "csv_finder": lambda: self._find_rllib_progress(run_dir, "ppo_lstm"),
                    "parser": parse_rllib_progress
                }
            }
            
            for name, config in algos.items():
                csv_path = config["csv_finder"]()
                history, reward, steps = config["parser"](csv_path)
                
                # Fetch memory stats if ParallelTrainer is active
                status = "PENDING"
                progress = 0
                current_iteration = 0
                total_iterations = 100
                logs = []
                
                if trainer:
                    status = trainer.status.get(name, "PENDING")
                    progress = trainer.progress.get(name, 0)
                    current_iteration = trainer.iterations.get(name, 0)
                    total_iterations = trainer.total_iterations.get(name, 100)
                    # Convert dashboard keys to match scripts keys if needed
                    key_map = {"Baseline": "Baseline", "PPO": "PPO", "DQN": "DQN", "PPO-LSTM": "PPO-LSTM"}
                    trainer_key = key_map.get(name, name)
                    if hasattr(trainer, "stdout_log") and trainer.stdout_log:
                        logs = trainer.stdout_log.get(trainer_key, [])
                
                response_data["algorithms"][name] = {
                    "status": status,
                    "progress": progress,
                    "current_iteration": current_iteration,
                    "total_iterations": total_iterations,
                    "current_reward": reward,
                    "current_steps": steps,
                    "history": history,
                    "logs": logs
                }
                
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        # 404 Not Found
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def _find_rllib_progress(self, run_dir, subfolder):
        pattern = os.path.join(run_dir, subfolder, "**", "progress.csv")
        files = glob.glob(pattern, recursive=True)
        if files:
            return max(files, key=os.path.getmtime)
        return None

def run_server():
    # Attempt to bind to 5000, increment if occupied
    port = 5000
    server = None
    while port < 5050:
        try:
            # Using ThreadingHTTPServer if available (Python 3.7+), otherwise HTTPServer
            if hasattr(http.server, "ThreadingHTTPServer"):
                server = http.server.ThreadingHTTPServer(("", port), DashboardHTTPRequestHandler)
            else:
                class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
                    pass
                server = ThreadingHTTPServer(("", port), DashboardHTTPRequestHandler)
            break
        except OSError:
            port += 1
            
    if server:
        DashboardState.port = port
        DashboardState.server_instance = server
        print(f"\n[DASHBOARD] Started on http://localhost:{port}\n")
        
        # Open in default browser after 1 second delay to ensure server is listening
        def open_browser():
            time.sleep(1.0)
            webbrowser.open(f"http://localhost:{port}")
            
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            server.serve_forever()
        except Exception:
            pass
    else:
        print("\n[DASHBOARD ERROR] Could not bind HTTP server to a local port.\n")

def start_dashboard(run_dir, timestamp):
    DashboardState.run_dir = run_dir
    DashboardState.timestamp = timestamp
    DashboardState.stage = "TRAINING"
    DashboardState.start_time = time.time()
    DashboardState.end_time = None
    
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    DashboardState.server_thread = server_thread

def stop_dashboard():
    if DashboardState.server_instance:
        DashboardState.server_instance.shutdown()
        DashboardState.server_instance.server_close()
        print("[DASHBOARD] Stopped server.")
