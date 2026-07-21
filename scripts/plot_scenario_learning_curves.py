import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Resolve project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Visual styling for publication-ready plots
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def smooth_data(y, window_size=15):
    """Apply moving average smoothing to curves"""
    if len(y) < window_size:
        return y
    return pd.Series(y).rolling(window=window_size, min_periods=1).mean().values

def load_rllib_progress(csv_path):
    """Load timesteps and rewards from RLlib progress.csv"""
    try:
        df = pd.read_csv(csv_path)
        # Handle alternate column names for different RLlib stacks
        reward_col = None
        for col in ["env_runners/episode_reward_mean", "episode_reward_mean"]:
            if col in df.columns:
                reward_col = col
                break
        
        if reward_col is None:
            return None, None
            
        steps = df["timesteps_total"].values
        rewards = df[reward_col].values
        
        # Handle nan values
        valid_idx = ~np.isnan(rewards)
        return steps[valid_idx], rewards[valid_idx]
    except Exception as e:
        print(f"Warning: Could not parse {csv_path}: {e}")
        return None, None

def load_qjc_progress(csv_path, bin_size=10, steps_per_episode=100):
    """Load and resample QJC training curve to match RLlib iteration steps"""
    try:
        df = pd.read_csv(csv_path)
        episodes = df["episode"].values
        rewards = df["total_reward"].values
        
        # Resample every 10 episodes (1000 steps)
        binned_steps = []
        binned_rewards = []
        
        for i in range(0, len(rewards), bin_size):
            chunk = rewards[i : i + bin_size]
            if len(chunk) == 0:
                continue
            avg_r = np.mean(chunk)
            last_ep = episodes[min(i + bin_size - 1, len(episodes) - 1)]
            steps = last_ep * steps_per_episode
            binned_steps.append(steps)
            binned_rewards.append(avg_r)
            
        return np.array(binned_steps), np.array(binned_rewards)
    except Exception as e:
        print(f"Warning: Could not parse QJC log {csv_path}: {e}")
        return None, None

def plot_scenario(scenario_dir, scenario_name, out_path):
    """Generate and save the reward learning curve for a single scenario"""
    print(f"Plotting learning curves for {scenario_name}...")
    
    # 1. Find CSV files
    ppo_csvs = glob.glob(os.path.join(scenario_dir, "ppo/**/progress.csv"), recursive=True)
    dqn_csvs = glob.glob(os.path.join(scenario_dir, "dqn/**/progress.csv"), recursive=True)
    ppo_lstm_csvs = glob.glob(os.path.join(scenario_dir, "ppo_lstm/**/progress.csv"), recursive=True)
    qjc_csv = os.path.join(scenario_dir, "qjc", "training_curve.csv")
    
    plt.figure(figsize=(9, 6))
    
    colors = {
        "PPO": "#1f77b4",       # Sleek Blue
        "DQN": "#ff7f0e",       # Vivid Orange
        "PPO-LSTM": "#9467bd",  # Elegant Purple
        "Baseline (QJC)": "#7f7f7f" # Soft Gray
    }
    
    max_steps_million = 0.0
    
    # Plot RLlib algos
    algo_mappings = [
        ("PPO", ppo_csvs, load_rllib_progress),
        ("DQN", dqn_csvs, load_rllib_progress),
        ("PPO-LSTM", ppo_lstm_csvs, load_rllib_progress),
    ]
    
    for label, csv_list, loader in algo_mappings:
        if csv_list:
            steps, rewards = loader(csv_list[0])
            if steps is not None:
                steps_million = steps / 1e6
                max_steps_million = max(max_steps_million, steps_million[-1])
                smoothed_rewards = smooth_data(rewards)
                
                plt.plot(steps_million, smoothed_rewards, label=label, 
                         color=colors[label], lw=2.5, alpha=0.9)
                # Scatter markers for key points
                plt.scatter(steps_million[::5], smoothed_rewards[::5], 
                            color=colors[label], s=15, alpha=0.7)
                
    # Plot Baseline QJC
    if os.path.exists(qjc_csv):
        steps, rewards = load_qjc_progress(qjc_csv)
        if steps is not None:
            steps_million = steps / 1e6
            smoothed_rewards = smooth_data(rewards, window_size=5)
            
            plt.plot(steps_million, smoothed_rewards, label="Baseline (QJC)", 
                     color=colors["Baseline (QJC)"], lw=2.0, alpha=0.8, linestyle="--")
            plt.scatter(steps_million[::3], smoothed_rewards[::3], 
                        color=colors["Baseline (QJC)"], s=10, alpha=0.6)

    plt.title(f"Training Progress: Reward Learning Curves ({scenario_name})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Environment Steps (Millions)", fontsize=12)
    plt.ylabel("Mean Episode Reward", fontsize=12)
    plt.xlim(0, max_steps_million if max_steps_million > 0 else 1.0)
    plt.grid(True)
    
    # Legend style matching academic papers
    plt.legend(loc="lower right", frameon=True, edgecolor="#e0e0e0", facecolor="white", shadow=False)
    
    # Cleanup spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot successfully to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Reward Learning Curves for completed scenarios")
    parser.add_argument("--run-dir", type=str, required=True, 
                        help="Directory of the cluster scenario run (e.g. artifacts/scenario_runs/2026-07-21_15-23-27)")
    args = parser.parse_args()
    
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.exists(run_dir):
        print(f"Error: Directory {run_dir} does not exist.")
        sys.exit(1)
        
    # Scan subfolders representing scenarios
    scenarios = ["S1-A", "S1-B", "S2-A", "S2-B"]
    for scen in scenarios:
        scen_path = os.path.join(run_dir, scen)
        if os.path.exists(scen_path):
            out_img = os.path.join(run_dir, f"learning_curve_{scen.lower()}.png")
            plot_scenario(scen_path, scen, out_img)
            
    print("\n" + "=" * 80)
    print("  ALL SCENARIO PLOTS GENERATED!")
    print(f"  Plots saved in: {run_dir}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
