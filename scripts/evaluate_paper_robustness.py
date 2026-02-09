
import os
import sys
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from simulation.controllers import UAVRuleBasedController
from confs.config import UAVConfig
from confs.env_config import EnvConfig
from confs.model_config import GlobalConfig, PPOLSTMConfig, QJCConfig
from core.logger import SimulationLogger

# Constants
SEEDS = range(100, 130) # 30 Seeds: 100 to 129
ALGOS = ["Baseline", "PPO", "DQN", "PPO-LSTM"]
METRICS = ["JSR", "Tracking_Acc", "Power", "SINR"]

def env_creator(config):
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

def find_latest_checkpoint(base_dir):
    import glob
    search_pattern = os.path.join(base_dir, "**", "checkpoint_*")
    ckpt_dirs = glob.glob(search_pattern, recursive=True)
    ckpt_dirs = [d for d in ckpt_dirs if os.path.isdir(d)]
    if not ckpt_dirs: return None
    return max(ckpt_dirs, key=os.path.getmtime)

def evaluate_algo(algo_name, run_dir):
    print(f"\n--- Evaluating {algo_name} ---")
    results = {m: [] for m in METRICS}
    
    # Setup Ray for RL algos
    algo_agent = None
    lstm_cell_size = 0
    
    if algo_name in ["PPO", "DQN", "PPO-LSTM"]:
        algo_dir = os.path.join(run_dir, algo_name.lower().replace("-", "_"))
        ckpt = find_latest_checkpoint(algo_dir)
        if not ckpt:
            print(f"Checkout not found for {algo_name} in {algo_dir}")
            return None
        
        try:
            # We assume Ray is initialized globally
            algo_agent = Algorithm.from_checkpoint(ckpt)
        except Exception as e:
            print(f"Error loading {algo_name}: {e}")
            return None
            
        if algo_name == "PPO-LSTM":
            lstm_cell_size = PPOLSTMConfig.LSTM_CELL_SIZE

    elif algo_name == "Baseline":
        # Baseline relies on loading within the loop or creating a dummy class
        # We'll handle inside the loop
        pass

    # Loop over seeds
    for seed in SEEDS:
        # print(f"Seed {seed}...", end="\r")
        
        # 1. Setup Env
        # No Logger to save speed/disk
        env = UAV_IoT_PZ_Env(logger=None, auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        
        # 2. Reset
        obs, infos = env.reset(seed=seed)
        
        # 3. Setup Agent logic
        lstm_state = [np.zeros(lstm_cell_size, dtype=np.float32), np.zeros(lstm_cell_size, dtype=np.float32)] if algo_name == "PPO-LSTM" else []
        
        if algo_name == "Baseline":
             env.attacker.load_model(os.path.join(run_dir, "baseline"))
             env.attacker.temp_xi = 0 # Greedy
        
        # 4. Run Episode
        terminated = False
        steps = 0
        
        # Episode Metrics
        ep_jammed = 0
        ep_reachable = 0
        ep_tracking = 0
        ep_power_sum = 0
        ep_sinr_sum = 0
        ep_sinr_count = 0
        
        while not terminated and steps < EnvConfig.MAX_STEPS:
            actions = {}
            steps += 1
            
            # --- Get Action ---
            jam_action = 0
            
            if algo_name == "Baseline":
                 jam_ch = env.attacker.select_channel_qjc()
                 jam_p = QJCConfig.MAX_POWER_LEVEL
                 jam_action = np.array([jam_ch, jam_p]) if not GlobalConfig.FLATTEN_ACTIONS else jam_ch * 10 + jam_p
            
            elif algo_agent:
                if "jammer_0" in obs:
                    jam_obs = obs["jammer_0"]
                    try:
                        # Compute Action
                        if algo_name == "PPO-LSTM":
                            res = algo_agent.compute_single_action(jam_obs, state=lstm_state, policy_id="jammer_policy", explore=False)
                            if isinstance(res, tuple):
                                jam_action = res[0]
                                lstm_state = res[1]
                            else:
                                jam_action = res
                        else:
                            # PPO / DQN
                            jam_action = algo_agent.compute_single_action(jam_obs, policy_id="jammer_policy", explore=False)
                    except:
                        jam_action = 0
                        
            actions["jammer_0"] = jam_action
            
            # Nodes
            for ag in env.agents:
                if "node" in ag: actions[ag] = 0
                
            # --- Step ---
            obs, rewards, terms, truncs, infos = env.step(actions)
            terminated = any(terms.values()) or any(truncs.values())
            
            # --- Collect Stats directly from Entities ---
            # Using Env internals for accuracy (Paper Metrics)
            
            # 1. JSR (Reachable)
            # We need to recalculate reachable count because step only returns 'jammed_count'
            # Let's peek into env structures
            uav_pos = env.uav.position
            # Recalculate reachable
            reachable_count = 0
            for node in env.nodes:
                d = np.linalg.norm(node.position - uav_pos)
                # Max range check (approx based on Path Loss and Tx Power)
                # If connected_theoretical is True (SINR_no_jam > Threshold)
                # We can re-use the internal logic if we access it, but easier to trust 'jammed_count' 
                # and assume 'reachable' is total nodes for now? 
                # NO. Paper says "N_reachable".
                # Let's iterate nodes and check connection_status
                # Status 0=Connected, 1=OutRange, 2=Jammed
                if node.connection_status != 1: # If not out of range
                    reachable_count += 1
            
            jammed_c = infos["jammer_0"]["jammed_count"]
            
            ep_jammed += jammed_c
            if reachable_count > 0:
                ep_reachable += reachable_count
            
            # 2. Tracking
            # Check directly
            if env.attacker.current_channel == env.uav.current_channel:
                ep_tracking += 1
                
            # 3. Power
            ep_power_sum += env.attacker.jamming_power # This is Output Power
            # Note: Paper says "Total Power" including Circuit? 
            # Controller calculates it. Let's use info["jammer_cost"] if available
            ep_power_sum += infos["jammer_0"]["jammer_cost"] # This is Total Power
            
            # 4. SINR
            # Average SINR of all nodes? 
            # Or average SINR of Target Node?
            # Let's take average SINR of all Reachable Nodes
            step_sinr_sum = 0
            step_sinr_n = 0
            for i in range(EnvConfig.NUM_NODES):
                # Retrieve from info
                s = infos["jammer_0"].get(f"node_{i}_sinr", 0)
                # Only include if reachable (status != 1) ?? 
                # The paper plot shows average SINR. Let's avg all nodes.
                step_sinr_sum += s
                step_sinr_n += 1
            
            if step_sinr_n > 0:
                ep_sinr_sum += (step_sinr_sum / step_sinr_n)
                ep_sinr_count += 1
                
        # --- End Episode ---
        
        # Calculate Episodes Averages
        val_jsr = (ep_jammed / ep_reachable) * 100 if ep_reachable > 0 else 0
        val_track = (ep_tracking / steps) * 100 if steps > 0 else 0
        val_power = ep_power_sum / steps if steps > 0 else 0
        val_sinr = ep_sinr_sum / ep_sinr_count if ep_sinr_count > 0 else 0
        
        results["JSR"].append(val_jsr)
        results["Tracking_Acc"].append(val_track)
        results["Power"].append(val_power)
        results["SINR"].append(val_sinr)
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to artifacts directory")
    args = parser.parse_args()
    
    # CRITICAL: Ray requires absolute paths for checkpoints to avoid "URI has empty scheme" errors
    run_dir_abs = os.path.abspath(args.run_dir)
    print(f"Using Absolute Run Directory: {run_dir_abs}")
    
    # Init Ray once
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    register_env("uav_iot_ppo_v1", env_creator)
    register_env("uav_iot_dqn_v1", env_creator)
    register_env("uav_iot_ppo_lstm_v1", env_creator)
    
    final_results = {}
    
    for algo in ALGOS:
        res = evaluate_algo(algo, run_dir_abs)
        if res:
            final_results[algo] = res
            
            # Print quick stats
            print(f"  > JSR: {np.mean(res['JSR']):.1f}% ± {np.std(res['JSR']):.1f}")
            print(f"  > Track: {np.mean(res['Tracking_Acc']):.1f}%")
            print(f"  > Power: {np.mean(res['Power']):.3f} W")
            print(f"  > SINR: {np.mean(res['SINR']):.2f} dB")
            
    ray.shutdown()
    
    # Save JSON
    out_file = os.path.join("paper", "robustness_results_30seeds.json")
    os.makedirs("paper", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\nSaved results to {out_file}")
    
    # Plotting
    plot_comparison(final_results)

def plot_comparison(results):
    import seaborn as sns
    sns.set_style("whitegrid")
    
    # Prepare Data
    means = {}
    stds = {}
    
    for m in METRICS:
        means[m] = []
        stds[m] = []
        for algo in ALGOS:
            if algo in results:
                means[m].append(np.mean(results[algo][m]))
                stds[m].append(np.std(results[algo][m]))
            else:
                means[m].append(0)
                stds[m].append(0)
            
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Algorithm Robustness Analysis (30 Random Seeds)", fontsize=20, fontweight='bold', y=0.98)
    
    colors = ['#7f8c8d', '#2ecc71', '#e74c3c', '#9b59b6'] # Baseline(Gray), PPO(Green), DQN(Red), LSTM(Purple)
    algo_labels = ["Baseline (QJC)", "PPO (Proposed)", "DQN", "PPO-LSTM"]
    
    # Helper for bar plots
    def plot_bar(ax, metric_key, title, ylabel, threshold=None):
        bars = ax.bar(algo_labels, means[metric_key], yerr=stds[metric_key], capsize=8, 
                     color=colors, alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='x', labelsize=11, rotation=15)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add values on top
        for bar, err in zip(bars, stds[metric_key]):
            height = bar.get_height()
            label = f"{height:.1f}" if abs(height) > 0.1 else f"{height:.3f}"
            ax.text(bar.get_x() + bar.get_width()/2., height + err + (height*0.05),
                    label, ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        if threshold is not None:
             ax.axhline(threshold, color='black', linestyle='--', linewidth=2, label="Threshold")
             ax.legend()

    # 1. JSR
    plot_bar(axes[0, 0], "JSR", "Jamming Success Rate (JSR)", "Success (%)")
    
    # 2. Tracking
    plot_bar(axes[0, 1], "Tracking_Acc", "Channel Tracking Accuracy", "Accuracy (%)")
    
    # 3. Power
    plot_bar(axes[1, 0], "Power", "Average Power Consumption", "Power (W)")
    
    # 4. SINR
    plot_bar(axes[1, 1], "SINR", "Average Network SINR", "SINR (dB)", threshold=0)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("paper/comparison_robustness.png", dpi=300, bbox_inches='tight')
    print("Saved plot to paper/comparison_robustness.png")


if __name__ == "__main__":
    main()
