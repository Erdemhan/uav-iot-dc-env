import os
import sys
# Force CPU mode for evaluation on Head Node to prevent CUDA Error 302
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Initialize Ray with runtime_env so remote worker nodes inherit working_dir and PYTHONPATH
if not ray.is_initialized():
    runtime_env = {
        "working_dir": project_root,
        "excludes": [".venv", "artifacts", "ray_results", "head-node-logs", ".git", "baseline_q_table", "node_modules"],
        "env_vars": {"PYTHONPATH": "."}
    }
    try:
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
    except Exception:
        pass

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from simulation.controllers import UAVRuleBasedController
from confs.config import UAVConfig
from confs.env_config import EnvConfig
from confs.model_config import GlobalConfig, PPOLSTMConfig, QJCConfig
from confs.opt_config import OptConfig

# Constants
SEEDS = OptConfig.EVAL_SEEDS

ALGOS = ["Baseline", "PPO", "DQN", "PPO-LSTM"]
METRICS = ["JSR", "Track_Reachable", "Power", "SINR", "Tracking_Acc", "Power_Gap", "Channel_Gap", "Power_Active", "Power_Idle"]

def env_creator(config):
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

def load_env_config_from_metadata(run_dir):
    """Override EnvConfig parameters using metadata.json to match training configuration."""
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"  [WARN] metadata.json not found: {meta_path}")
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ec = meta.get("env_config", {})
    from confs.env_config import EnvConfig
    if "NUM_NODES"         in ec: EnvConfig.NUM_NODES        = ec["NUM_NODES"]
    if "NUM_UAVS"          in ec: EnvConfig.NUM_UAVS         = ec["NUM_UAVS"]
    if "AREA_SIZE"         in ec: EnvConfig.AREA_SIZE        = ec["AREA_SIZE"]
    if "ATTACKER_POS_X"    in ec: EnvConfig.ATTACKER_POS_X   = ec["ATTACKER_POS_X"]
    if "ATTACKER_POS_Y"    in ec: EnvConfig.ATTACKER_POS_Y   = ec["ATTACKER_POS_Y"]
    if "MAX_JAMMING_POWER" in ec: EnvConfig.MAX_JAMMING_POWER = ec["MAX_JAMMING_POWER"]
    if "P_TX_NODE"         in ec: EnvConfig.P_TX_NODE        = ec["P_TX_NODE"]
    if "P_TX_UAV"          in ec: EnvConfig.P_TX_UAV         = ec["P_TX_UAV"]
    if "MAX_STEPS"         in ec: EnvConfig.MAX_STEPS        = ec["MAX_STEPS"]
    if "STEP_TIME"         in ec: EnvConfig.STEP_TIME        = ec["STEP_TIME"]
    if "UAV_START_X"       in ec: EnvConfig.UAV_START_X      = ec["UAV_START_X"]
    if "UAV_START_Y"       in ec: EnvConfig.UAV_START_Y      = ec["UAV_START_Y"]
    if "UAV_START_Z"       in ec: EnvConfig.UAV_START_Z      = ec["UAV_START_Z"]
    if "UAV_SPEED"         in ec: EnvConfig.UAV_SPEED        = ec["UAV_SPEED"]
    if "W_SUCCESS"         in ec: EnvConfig.W_SUCCESS        = ec["W_SUCCESS"]
    if "W_TRACKING"        in ec: EnvConfig.W_TRACKING       = ec["W_TRACKING"]
    if "W_COST"            in ec: EnvConfig.W_COST           = ec["W_COST"]
    obs_dim = EnvConfig.get_obs_dim()
    print(f"  EnvConfig override completed. NUM_NODES={EnvConfig.NUM_NODES}, obs_dim={obs_dim}")

def find_latest_checkpoint(base_dir):
    import glob
    search_pattern = os.path.join(base_dir, "**", "checkpoint_*")
    ckpt_dirs = glob.glob(search_pattern, recursive=True)
    ckpt_dirs = [d for d in ckpt_dirs if os.path.isdir(d)]
    if not ckpt_dirs: return None
    return max(ckpt_dirs, key=os.path.getmtime)

@ray.remote(num_gpus=1)
def evaluate_algo_remote(algo_name, checkpoint_bytes, baseline_bytes, env_cfg_dict):
    """Executes evaluation loop on a Ray Cluster Worker PC with GPU access."""
    import os
    import sys
    import numpy as np
    import copy
    from ray.tune.registry import register_env
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.rllib.algorithms.algorithm import Algorithm

    # Apply env_cfg_dict dynamically on Worker PC
    from confs.env_config import EnvConfig
    from confs.model_config import GlobalConfig, PPOLSTMConfig, QJCConfig
    from confs.opt_config import OptConfig
    from simulation.pettingzoo_env import UAV_IoT_PZ_Env

    if "NUM_NODES" in env_cfg_dict: EnvConfig.NUM_NODES = env_cfg_dict["NUM_NODES"]
    if "NUM_UAVS" in env_cfg_dict: EnvConfig.NUM_UAVS = env_cfg_dict["NUM_UAVS"]
    if "AREA_SIZE" in env_cfg_dict: EnvConfig.AREA_SIZE = env_cfg_dict["AREA_SIZE"]
    if "ATTACKER_POS_X" in env_cfg_dict: EnvConfig.ATTACKER_POS_X = env_cfg_dict["ATTACKER_POS_X"]
    if "ATTACKER_POS_Y" in env_cfg_dict: EnvConfig.ATTACKER_POS_Y = env_cfg_dict["ATTACKER_POS_Y"]
    if "MAX_JAMMING_POWER" in env_cfg_dict: EnvConfig.MAX_JAMMING_POWER = env_cfg_dict["MAX_JAMMING_POWER"]
    if "P_TX_NODE" in env_cfg_dict: EnvConfig.P_TX_NODE = env_cfg_dict["P_TX_NODE"]
    if "P_TX_UAV" in env_cfg_dict: EnvConfig.P_TX_UAV = env_cfg_dict["P_TX_UAV"]
    if "MAX_STEPS" in env_cfg_dict: EnvConfig.MAX_STEPS = env_cfg_dict["MAX_STEPS"]
    if "STEP_TIME" in env_cfg_dict: EnvConfig.STEP_TIME = env_cfg_dict["STEP_TIME"]
    if "UAV_START_X" in env_cfg_dict: EnvConfig.UAV_START_X = env_cfg_dict["UAV_START_X"]
    if "UAV_START_Y" in env_cfg_dict: EnvConfig.UAV_START_Y = env_cfg_dict["UAV_START_Y"]
    if "UAV_START_Z" in env_cfg_dict: EnvConfig.UAV_START_Z = env_cfg_dict["UAV_START_Z"]
    if "UAV_SPEED" in env_cfg_dict: EnvConfig.UAV_SPEED = env_cfg_dict["UAV_SPEED"]
    if "W_SUCCESS" in env_cfg_dict: EnvConfig.W_SUCCESS = env_cfg_dict["W_SUCCESS"]
    if "W_TRACKING" in env_cfg_dict: EnvConfig.W_TRACKING = env_cfg_dict["W_TRACKING"]
    if "W_COST" in env_cfg_dict: EnvConfig.W_COST = env_cfg_dict["W_COST"]

    def eval_env_creator(cfg):
        return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

    for name in ["uav_iot_ppo_v1", "uav_iot_dqn_v1", "uav_iot_ppo_lstm_v1", 
                "uav_iot_ppo_gpu_v1", "uav_iot_dqn_gpu_v1", "uav_iot_ppo_lstm_gpu_v1"]:
        try:
            register_env(name, eval_env_creator)
        except Exception:
            pass

    results = {m: [] for m in METRICS}
    algo_agent = None
    lstm_cell_size = 0

    if checkpoint_bytes:
        tmp_dir = f"/tmp/eval_ckpt_{algo_name.lower().replace('-', '_')}"
        if os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        for rel_path, data in checkpoint_bytes.items():
            dest = os.path.join(tmp_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fp:
                fp.write(data)

        try:
            algo_agent = Algorithm.from_checkpoint(tmp_dir)
        except Exception as e:
            print(f"Error loading {algo_name} on Worker PC: {e}")
            return None

        if algo_name == "PPO-LSTM":
            lstm_cell_size = PPOLSTMConfig.LSTM_CELL_SIZE

    tmp_baseline_dir = None
    if baseline_bytes:
        tmp_baseline_dir = f"/tmp/baseline_{algo_name.lower()}"
        os.makedirs(tmp_baseline_dir, exist_ok=True)
        for rel_path, data in baseline_bytes.items():
            dest = os.path.join(tmp_baseline_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fp:
                fp.write(data)

    seeds = OptConfig.EVAL_SEEDS
    for seed in seeds:
        env = UAV_IoT_PZ_Env(logger=None, auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        obs, infos = env.reset(seed=seed)
        lstm_state = [np.zeros(lstm_cell_size, dtype=np.float32), np.zeros(lstm_cell_size, dtype=np.float32)] if algo_name == "PPO-LSTM" else []

        if algo_name == "Baseline" and tmp_baseline_dir:
            env.attacker.load_model(tmp_baseline_dir)
            env.attacker.temp_xi = 0
            env.attacker.channel_counts = np.zeros(env.attacker.num_channels)

        terminated = False
        steps = 0
        ep_jammed = 0
        ep_reachable = 0
        ep_tracking_all = 0
        ep_tracking_reachable = 0
        ep_power_sum = 0
        ep_power_active_sum = 0
        ep_power_active_steps = 0
        ep_power_idle_sum = 0
        ep_power_idle_steps = 0
        ep_sinr_sum = 0
        ep_sinr_count = 0

        while not terminated and steps < EnvConfig.MAX_STEPS:
            actions = {}
            steps += 1
            jam_action = 0

            if algo_name == "Baseline":
                jam_ch = env.attacker.select_channel_qjc()
                jam_p = QJCConfig.MAX_POWER_LEVEL
                jam_action = jam_ch * 10 + jam_p if GlobalConfig.FLATTEN_ACTIONS else np.array([jam_ch, jam_p])
            elif algo_agent:
                if "jammer_0" in obs:
                    jam_obs = obs["jammer_0"]
                    try:
                        if algo_name == "PPO-LSTM":
                            res = algo_agent.compute_single_action(jam_obs, state=lstm_state, policy_id="jammer_policy", explore=False)
                            if isinstance(res, tuple):
                                jam_action = res[0]
                                lstm_state = res[1]
                            else:
                                jam_action = res
                        else:
                            jam_action = algo_agent.compute_single_action(jam_obs, policy_id="jammer_policy", explore=False)
                    except Exception:
                        jam_action = 0

            actions["jammer_0"] = jam_action
            for ag in env.agents:
                if "node" in ag: actions[ag] = 0

            obs, rewards, terms, truncs, infos = env.step(actions)
            terminated = any(terms.values()) or any(truncs.values())

            if algo_name == "Baseline":
                r_jam = rewards.get("jammer_0", 0)
                env.attacker.update_qjc(jam_ch, r_jam)

            reachable_count = sum(1 for n in env.nodes if n.connection_status != 1)
            jammed_c = infos["jammer_0"]["jammed_count"]
            ep_jammed += jammed_c
            if reachable_count > 0:
                ep_reachable += reachable_count

            closest_uav = min(env.uavs, key=lambda uav: np.linalg.norm(uav.position - env.attacker.position))
            ch_match = (env.attacker.current_channel == closest_uav.current_channel)
            if ch_match:
                ep_tracking_all += 1
                if reachable_count > 0:
                    ep_tracking_reachable += 1

            current_power = infos["jammer_0"]["jammer_cost"]
            ep_power_sum += current_power
            if reachable_count > 0:
                ep_power_active_sum += current_power
                ep_power_active_steps += 1
            else:
                ep_power_idle_sum += current_power
                ep_power_idle_steps += 1

            step_sinr_sum = 0
            step_sinr_n = 0
            for i in range(EnvConfig.NUM_NODES):
                s = infos["jammer_0"].get(f"node_{i}_sinr", 0)
                step_sinr_sum += s
                step_sinr_n += 1
            if step_sinr_n > 0:
                step_avg_sinr = step_sinr_sum / step_sinr_n
                step_avg_sinr_db = 10 * np.log10(max(step_avg_sinr, 1e-12))
                ep_sinr_sum += step_avg_sinr_db
                ep_sinr_count += 1

        val_jsr = (ep_jammed / ep_reachable) * 100 if ep_reachable > 0 else 0.0
        val_track_all = (ep_tracking_all / steps) * 100 if steps > 0 else 0.0
        val_track_reach = (ep_tracking_reachable / ep_reachable) * 100 if ep_reachable > 0 else 0.0
        val_power_gap = val_track_reach - val_jsr
        val_channel_gap = 100.0 - val_track_reach
        val_power = ep_power_sum / steps if steps > 0 else 0.0
        val_power_active = ep_power_active_sum / ep_power_active_steps if ep_power_active_steps > 0 else 0.0
        val_power_idle = ep_power_idle_sum / ep_power_idle_steps if ep_power_idle_steps > 0 else 0.0
        val_sinr = ep_sinr_sum / ep_sinr_count if ep_sinr_count > 0 else 0.0

        results["JSR"].append(val_jsr)
        results["Tracking_Acc"].append(val_track_all)
        results["Track_Reachable"].append(val_track_reach)
        results["Power_Gap"].append(val_power_gap)
        results["Channel_Gap"].append(val_channel_gap)
        results["Power"].append(val_power)
        results["Power_Active"].append(val_power_active)
        results["Power_Idle"].append(val_power_idle)
        results["SINR"].append(val_sinr)

    return results

def main():
    from core.logger import setup_console_logging
    setup_console_logging("evaluate_paper_robustness")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to artifacts directory")
    args = parser.parse_args()
    
    run_dir_abs = os.path.abspath(args.run_dir)
    print(f"Using Absolute Run Directory: {run_dir_abs}")
    
    # Load env config from metadata
    load_env_config_from_metadata(run_dir_abs)
    
    # Init Ray with full runtime_env including working_dir
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not ray.is_initialized():
        runtime_env = {
            "working_dir": project_root,
            "excludes": [".venv", "artifacts", "ray_results", "head-node-logs", ".git", "baseline_q_table", "node_modules"],
            "env_vars": {"PYTHONPATH": "."}
        }
        try:
            ray.init(
                address="auto",
                ignore_reinit_error=True,
                log_to_driver=False,
                runtime_env=runtime_env
            )
        except Exception:
            pass
        
    for name in ["uav_iot_ppo_v1", "uav_iot_dqn_v1", "uav_iot_ppo_lstm_v1", 
                "uav_iot_ppo_gpu_v1", "uav_iot_dqn_gpu_v1", "uav_iot_ppo_lstm_gpu_v1"]:
        try:
            register_env(name, env_creator)
        except Exception:
            pass
    
    # Extract env_cfg_dict from EnvConfig
    env_cfg_dict = {
        "NUM_NODES": EnvConfig.NUM_NODES,
        "NUM_UAVS": EnvConfig.NUM_UAVS,
        "AREA_SIZE": EnvConfig.AREA_SIZE,
        "ATTACKER_POS_X": EnvConfig.ATTACKER_POS_X,
        "ATTACKER_POS_Y": EnvConfig.ATTACKER_POS_Y,
        "MAX_JAMMING_POWER": EnvConfig.MAX_JAMMING_POWER,
        "P_TX_NODE": EnvConfig.P_TX_NODE,
        "P_TX_UAV": EnvConfig.P_TX_UAV,
        "MAX_STEPS": EnvConfig.MAX_STEPS,
        "STEP_TIME": EnvConfig.STEP_TIME,
        "UAV_START_X": EnvConfig.UAV_START_X,
        "UAV_START_Y": EnvConfig.UAV_START_Y,
        "UAV_START_Z": EnvConfig.UAV_START_Z,
        "UAV_SPEED": EnvConfig.UAV_SPEED,
        "W_SUCCESS": EnvConfig.W_SUCCESS,
        "W_TRACKING": EnvConfig.W_TRACKING,
        "W_COST": EnvConfig.W_COST,
    }

    final_results = {}
    pending_tasks = {}

    for algo in ALGOS:
        print(f"\n---> Dispatching Remote Evaluation for {algo} to Worker PC (num_gpus=1)...")
        checkpoint_bytes = None
        baseline_bytes = None

        if algo in ["PPO", "DQN", "PPO-LSTM"]:
            algo_dir = os.path.join(run_dir_abs, algo.lower().replace("-", "_"))
            ckpt_dir = find_latest_checkpoint(algo_dir)
            if ckpt_dir and os.path.exists(ckpt_dir):
                checkpoint_bytes = {}
                for root, dirs, files in os.walk(ckpt_dir):
                    for f in files:
                        full_p = os.path.join(root, f)
                        rel_p = os.path.relpath(full_p, ckpt_dir)
                        with open(full_p, "rb") as fp:
                            checkpoint_bytes[rel_p] = fp.read()
            else:
                print(f"  [WARN] Checkpoint not found for {algo} in {algo_dir}")
                continue

        elif algo == "Baseline":
            base_dir = os.path.join(run_dir_abs, "baseline")
            if os.path.exists(base_dir):
                baseline_bytes = {}
                for root, dirs, files in os.walk(base_dir):
                    for f in files:
                        full_p = os.path.join(root, f)
                        rel_p = os.path.relpath(full_p, base_dir)
                        with open(full_p, "rb") as fp:
                            baseline_bytes[rel_p] = fp.read()

        future = evaluate_algo_remote.options(num_gpus=1).remote(algo, checkpoint_bytes, baseline_bytes, env_cfg_dict)
        pending_tasks[algo] = future

    # Gather results on Head Node
    for algo, future in pending_tasks.items():
        try:
            print(f"\n--- Gathering Evaluation Results for {algo} from Worker PC ---")
            res = ray.get(future)
            if res:
                final_results[algo] = res
                print(f"  > JSR: {np.mean(res['JSR']):.1f}% ± {np.std(res['JSR']):.1f}")
                print(f"  > Track (All Steps): {np.mean(res['Tracking_Acc']):.1f}%")
                print(f"  > Track (Reachable): {np.mean(res['Track_Reachable']):.1f}%")
                print(f"  > Power Gap: {np.mean(res['Power_Gap']):.1f}%")
                print(f"  > Channel Gap: {np.mean(res['Channel_Gap']):.1f}%")
                print(f"  > Power (Overall Avg): {np.mean(res['Power']):.3f} W")
                print(f"  > Power (Active Collection): {np.mean(res['Power_Active']):.3f} W")
                print(f"  > Power (Idle / Transit): {np.mean(res['Power_Idle']):.3f} W")
                print(f"  > SINR: {np.mean(res['SINR']):.2f} dB")
        except Exception as e:
    comparison_dir = os.path.join(run_dir_abs, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Plotting on Head Node
    plot_comparison(final_results, comparison_dir)
    plot_loss_decomposition(final_results, comparison_dir)
    
    # Save JSON on Head Node
    out_file = os.path.join(comparison_dir, "robustness_results_30seeds.json")
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\nSaved results and plots to {comparison_dir}")

    try:
        os._exit(0)
    except Exception:
        pass

def plot_comparison(results, output_dir):
    import seaborn as sns
    sns.set_style("whitegrid")
    
    # Metrics list
    plot_metrics = ["JSR", "Track_Reachable", "Power_Gap", "Channel_Gap", "Power"]
    metric_titles = {
        "JSR": "Jamming Success Rate (JSR)",
        "Track_Reachable": "Channel Tracking Accuracy",
        "Power_Gap": "Power Loss (Power Gap)",
        "Channel_Gap": "Channel Loss (Channel Gap)",
        "Power": "Overall Average Power"
    }
    metric_ylabels = {
        "JSR": "Success (%)",
        "Track_Reachable": "Accuracy (%)",
        "Power_Gap": "Loss (%)",
        "Channel_Gap": "Loss (%)",
        "Power": "Power (W)"
    }
    
    means = {}
    stds = {}
    
    for m in plot_metrics:
        means[m] = []
        stds[m] = []
        for algo in ALGOS:
            if algo in results:
                means[m].append(np.mean(results[algo][m]))
                stds[m].append(np.std(results[algo][m]))
            else:
                means[m].append(0)
                stds[m].append(0)
            
    # Plot 2x3 Grid
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(f"Algorithm Robustness Analysis (30 Random Seeds)", fontsize=22, fontweight='bold', y=0.98)
    
    colors = ['#7f8c8d', '#2ecc71', '#e74c3c', '#9b59b6'] # Baseline(Gray), PPO(Green), DQN(Red), LSTM(Purple)
    algo_labels = ["Baseline (QJC)", "PPO (Proposed)", "DQN", "PPO-LSTM"]
    
    # Helper for bar plots
    def plot_bar(ax, metric_key, title, ylabel):
        bars = ax.bar(algo_labels, means[metric_key], yerr=stds[metric_key], capsize=8, 
                     color=colors, alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='x', labelsize=11, rotation=15)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add values on top
        for bar, err in zip(bars, stds[metric_key]):
            height = bar.get_height()
            label = f"{height:.3f}" if metric_key == "Power" else f"{height:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2., height + err + (height*0.02 + 0.002 if metric_key == "Power" else height*0.02),
                    label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Draw individual bar plots
    plot_bar(axes[0, 0], "JSR", metric_titles["JSR"], metric_ylabels["JSR"])
    plot_bar(axes[0, 1], "Track_Reachable", metric_titles["Track_Reachable"], metric_ylabels["Track_Reachable"])
    plot_bar(axes[0, 2], "Power", metric_titles["Power"], metric_ylabels["Power"])
    
    plot_bar(axes[1, 0], "Power_Gap", metric_titles["Power_Gap"], metric_ylabels["Power_Gap"])
    plot_bar(axes[1, 1], "Channel_Gap", metric_titles["Channel_Gap"], metric_ylabels["Channel_Gap"])
    
    # Active vs Idle Power Grouped Bar Chart (bottom right)
    ax_power = axes[1, 2]
    x_indices = np.arange(len(ALGOS))
    bar_width = 0.35
    
    active_means = []
    active_stds = []
    idle_means = []
    idle_stds = []
    
    for algo in ALGOS:
        if algo in results:
            active_means.append(np.mean(results[algo]["Power_Active"]))
            active_stds.append(np.std(results[algo]["Power_Active"]))
            idle_means.append(np.mean(results[algo]["Power_Idle"]))
            idle_stds.append(np.std(results[algo]["Power_Idle"]))
        else:
            active_means.append(0)
            active_stds.append(0)
            idle_means.append(0)
            idle_stds.append(0)
            
    bars_active = ax_power.bar(x_indices - bar_width/2, active_means, bar_width, yerr=active_stds, capsize=4,
                         color="#e74c3c", alpha=0.9, edgecolor='black', linewidth=1.2, label="Active Collection")
    bars_idle = ax_power.bar(x_indices + bar_width/2, idle_means, bar_width, yerr=idle_stds, capsize=4,
                       color="#3498db", alpha=0.9, edgecolor='black', linewidth=1.2, label="Idle / Transit")
                       
    ax_power.set_title("Active vs. Idle Power Comparison", fontsize=14, fontweight='bold', pad=15)
    ax_power.set_ylabel("Power (W)", fontsize=12)
    ax_power.set_xticks(x_indices)
    ax_power.set_xticklabels(algo_labels, fontsize=11, rotation=15)
    ax_power.legend(fontsize=10, loc="upper right")
    ax_power.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add values on top of bars
    for bar in bars_active:
        h = bar.get_height()
        ax_power.text(bar.get_x() + bar.get_width()/2., h + 0.005, f"{h:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars_idle:
        h = bar.get_height()
        ax_power.text(bar.get_x() + bar.get_width()/2., h + 0.005, f"{h:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_plot = os.path.join(output_dir, "comparison_robustness.png")
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_plot}")
    plt.close()

def plot_loss_decomposition(results, output_dir):
    # Stacked bar plot for JSR, Power Gap, and Channel Gap
    # Total sums to 100%
    fig, ax = plt.subplots(figsize=(9, 6))
    
    jsr_m = [np.mean(results[a]["JSR"]) if a in results else 0.0 for a in ALGOS]
    pwr_m = [np.mean(results[a]["Power_Gap"]) if a in results else 0.0 for a in ALGOS]
    ch_m  = [np.mean(results[a]["Channel_Gap"]) if a in results else 0.0 for a in ALGOS]
    jsr_s = [np.std(results[a]["JSR"]) if a in results else 0.0 for a in ALGOS]
    
    x = np.arange(len(ALGOS))
    w = 0.55
    
    C_JSR = "#2ecc71" # Green
    C_PWR = "#e67e22" # Orange
    C_CH  = "#e74c3c" # Red
    ALPHA = 0.90
    
    # Stacked bars
    ax.bar(x, ch_m, w, color=C_CH, alpha=ALPHA, label="Channel Loss")
    ax.bar(x, pwr_m, w, color=C_PWR, alpha=ALPHA, label="Power Loss", bottom=ch_m)
    bottom_jsr = [c + p for c, p in zip(ch_m, pwr_m)]
    ax.bar(x, jsr_m, w, color=C_JSR, alpha=ALPHA, label="JSR (Success)", bottom=bottom_jsr)
    
    # Error bars for JSR
    ax.errorbar(x, [b + j for b, j in zip(bottom_jsr, jsr_m)],
                yerr=jsr_s, fmt="none", color="#1a8a4a", capsize=5, lw=1.5, capthick=1.5)
                
    # Labels
    for i, (c, p, j, js) in enumerate(zip(ch_m, pwr_m, jsr_m, jsr_s)):
        if c >= 5:
            ax.text(x[i], c / 2, f"{c:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        if p >= 5:
            ax.text(x[i], c + p / 2, f"{p:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        if j >= 5:
            ax.text(x[i], c + p + j / 2, f"{j:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        top = c + p + j
        ax.text(x[i], top + 2.5, f"{j:.1f} ± {js:.1f}%", ha="center", va="bottom", fontsize=9, color="#1a8a4a", fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels(["QJC\n(Baseline)", "PPO", "DQN", "PPO-LSTM"], fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage of Reachable Steps (%)", fontsize=12)
    ax.set_title("Reachable-Normalized Loss Decomposition", fontsize=14, fontweight="bold", pad=15)
    ax.axhline(100, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    
    handles = [
        mpatches.Patch(color=C_JSR, alpha=ALPHA, label="JSR (Jamming Success Rate)"),
        mpatches.Patch(color=C_PWR, alpha=ALPHA, label="Power Loss (correct channel, distance/power too large)"),
        mpatches.Patch(color=C_CH,  alpha=ALPHA, label="Channel Loss (wrong channel)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    out_plot = os.path.join(output_dir, "loss_decomposition.png")
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out_plot}")
    plt.close()

if __name__ == "__main__":
    main()
