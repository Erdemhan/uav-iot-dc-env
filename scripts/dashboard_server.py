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
    target_page = "index.html"

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
def get_project_root():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(base_dir)

def get_active_run_dir():
    """Dynamically get the active run directory. Supports both training runs and Optuna tuning runs."""
    project_root = get_project_root()
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
        import urllib.parse
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)
        run_param = query_params.get("run", [None])[0]
        
        project_root = get_project_root()
        if run_param:
            # Normalize path delimiters for windows/linux and prevent directory traversal
            clean_run_param = run_param.replace("\\", "/").strip("/")
            resolved_run_dir = os.path.abspath(os.path.join(project_root, "artifacts", clean_run_param))
            artifacts_dir = os.path.abspath(os.path.join(project_root, "artifacts"))
            
            # Use normcase to handle drive letter case mismatch (c: vs C:) on Windows
            if os.path.normcase(resolved_run_dir).startswith(os.path.normcase(artifacts_dir)) and os.path.exists(resolved_run_dir):
                run_dir = resolved_run_dir
            else:
                run_dir = get_active_run_dir()
        else:
            run_dir = get_active_run_dir()

        # Serve list of runs
        if path == "/api/list_runs":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            
            runs = []
            search_paths = [
                ("training", os.path.join(project_root, "artifacts", "training")),
                ("tune", os.path.join(project_root, "artifacts", "tune")),
                ("legacy", os.path.join(project_root, "artifacts"))
            ]
            
            seen_ids = set()
            for run_group, folder_path in search_paths:
                if not os.path.isdir(folder_path):
                    continue
                for item in os.listdir(folder_path):
                    if run_group == "legacy" and item in ["training", "tune"]:
                        continue
                    item_path = os.path.join(folder_path, item)
                    if not os.path.isdir(item_path):
                        continue
                        
                    run_id = item
                    if run_group != "legacy":
                        run_id = f"{run_group}/{item}"
                        
                    if run_id in seen_ids:
                        continue
                        
                    meta_path = os.path.join(item_path, "metadata.json")
                    status_path = os.path.join(item_path, "status.json")
                    if os.path.exists(meta_path) or os.path.exists(status_path):
                        run_type = "training"
                        algo_name = ""
                        if os.path.exists(meta_path):
                            try:
                                with open(meta_path, "r", encoding="utf-8") as f:
                                    m = json.load(f)
                                    algo_name = m.get("algo", "")
                                    phase = m.get("phase", None)
                                    if phase == 1:
                                        run_type = "hpo"
                                    elif phase == 2:
                                        run_type = "reward"
                            except:
                                pass
                        
                        mtime = os.path.getmtime(item_path)
                        runs.append({
                            "id": run_id,
                            "type": run_type,
                            "algo": algo_name,
                            "mtime": mtime,
                            "date_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                        })
                        seen_ids.add(run_id)
            # Sort by mtime descending
            runs.sort(key=lambda x: x["mtime"], reverse=True)
            self.wfile.write(json.dumps({"runs": runs}).encode("utf-8"))
            return

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
                        elapsed = last.get("time_total_s", None)
                        if elapsed not in (None, ""):
                            elapsed = float(elapsed)
                        else:
                            elapsed = 0.0

                        active_trials.append({
                            "trial_id": trial_folder,
                            "current_iteration": current_iter,
                            "total_iterations": total_iters,
                            "objective": objective,
                            "jsr": jsr,
                            "num_rows": len(rows),
                            "duration_seconds": elapsed
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
            else:
                # Dynamic scan fallback to show running/completed trials before optuna_trials.json is saved
                tune_results_dir = os.path.join(run_dir, "tune_results", "optuna_study")
                if os.path.isdir(tune_results_dir):
                    trials_list = []
                    best_val = -float("inf")
                    best_num = None
                    
                    try:
                        # Sort folders by creation/mtime so trial indices are consistent
                        folders = sorted(os.listdir(tune_results_dir), key=lambda x: os.path.getmtime(os.path.join(tune_results_dir, x)))
                        for idx, folder in enumerate(folders):
                            trial_path = os.path.join(tune_results_dir, folder)
                            if not os.path.isdir(trial_path):
                                continue
                            progress_csv = os.path.join(trial_path, "progress.csv")
                            params_path = os.path.join(trial_path, "params.json")
                            
                            if not os.path.exists(progress_csv):
                                continue
                                
                            try:
                                params = {}
                                if os.path.exists(params_path):
                                    with open(params_path, "r", encoding="utf-8") as pf:
                                        params = json.load(pf)
                                
                                with open(progress_csv, "r", encoding="utf-8") as f:
                                    reader = csv.DictReader(f)
                                    rows = list(reader)
                                if not rows:
                                    continue
                                
                                last = rows[-1]
                                objective = last.get("objective", None)
                                if objective not in (None, ""):
                                    objective = float(objective)
                                else:
                                    objective = None
                                    
                                elapsed = last.get("time_total_s", None)
                                if elapsed not in (None, ""):
                                    elapsed = float(elapsed)
                                else:
                                    elapsed = 0.0
                                    
                                cur_iter = int(float(last.get("training_iteration", 0) or 0))
                                total_iters = params.get("iterations", 100) if isinstance(params, dict) else 100
                                
                                display_params = {}
                                skip_keys = ["algo", "iterations", "phase", "num_workers", "num_gpus", "env_config"]
                                if isinstance(params, dict):
                                    for k, v in params.items():
                                        if k not in skip_keys:
                                            display_params[k] = v
                                            
                                is_done = last.get("done", "False") == "True" or cur_iter >= total_iters
                                state = "TrialState.COMPLETE" if is_done else "TrialState.RUNNING"
                                
                                trials_list.append({
                                    "number": idx,
                                    "value": objective,
                                    "state": state,
                                    "params": display_params,
                                    "duration_seconds": elapsed
                                })
                                
                                if objective is not None and objective > best_val:
                                    best_val = objective
                                    best_num = idx
                            except Exception:
                                pass
                        
                        response_data["trials"] = trials_list
                        if best_num is not None:
                            response_data["best_value"] = best_val
                            response_data["best_trial_number"] = best_num
                    except Exception:
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
            
            # Read use_gpu from GlobalConfig
            use_gpu = False
            try:
                from confs.model_config import GlobalConfig
                use_gpu = GlobalConfig.USE_GPU
            except Exception:
                pass

            # Gather progress data
            response_data = {
                "timestamp": DashboardState.timestamp,
                "stage": DashboardState.stage,
                "elapsed_time": elapsed,
                "use_gpu": use_gpu,
                "algorithms": {}
            }
            
            # Read status.json from disk first (cross-process sync)
            disk_status = None
            status_file = os.path.join(run_dir, "status.json")
            if os.path.exists(status_file):
                try:
                    with open(status_file, "r", encoding="utf-8") as sf:
                        disk_status = json.load(sf)
                except Exception:
                    pass
            
            if disk_status:
                response_data["stage"] = disk_status.get("stage", response_data["stage"])
            
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
                
                status = "PENDING"
                progress = 0
                current_iteration = 0
                total_iterations = 100
                logs = []
                
                if disk_status and "algorithms" in disk_status and name in disk_status["algorithms"]:
                    algo_disk = disk_status["algorithms"][name]
                    status = algo_disk.get("status", "PENDING")
                    progress = algo_disk.get("progress", 0)
                    current_iteration = algo_disk.get("current_iteration", 0)
                    total_iterations = algo_disk.get("total_iterations", 100)
                    logs = algo_disk.get("logs", [])
                elif trainer:
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
            target = f"http://localhost:{port}"
            if DashboardState.target_page and DashboardState.target_page != "index.html":
                target += "/" + DashboardState.target_page
            webbrowser.open(target)
            
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            server.serve_forever()
        except Exception:
            pass
    else:
        print("\n[DASHBOARD ERROR] Could not bind HTTP server to a local port.\n")

def start_dashboard(run_dir, timestamp, page="index.html"):
    DashboardState.run_dir = run_dir
    DashboardState.timestamp = timestamp
    DashboardState.target_page = page
    DashboardState.stage = "TRAINING"
    DashboardState.start_time = time.time()
    DashboardState.end_time = None
    
    # Update active run file so get_active_run_dir() resolves to the correct path
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        active_file = os.path.join(project_root, "dashboard_active_run.txt")
        # Ensure path is absolute and clean
        abs_run_dir = os.path.abspath(run_dir)
        with open(active_file, "w", encoding="utf-8") as f:
            f.write(abs_run_dir)
        print(f"[DASHBOARD] Updated active run directory to: {abs_run_dir}")
    except Exception as e:
        print(f"[DASHBOARD] Warning: Could not write dashboard_active_run.txt: {e}")
    
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    DashboardState.server_thread = server_thread

def stop_dashboard():
    if DashboardState.server_instance:
        DashboardState.server_instance.shutdown()
        DashboardState.server_instance.server_close()
        print("[DASHBOARD] Stopped server.")
