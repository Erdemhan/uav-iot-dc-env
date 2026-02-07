
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(exp_name):
    path = os.path.join("experiments", exp_name, "history.csv")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    df = pd.read_csv(path)
    return df

def preprocess_df(df):
    """Adds missing columns or derived metrics."""
    # SINR dB
    sinr_cols = [c for c in df.columns if "node_" in c and "_sinr" in c]
    if sinr_cols:
        lin_sinr = df[sinr_cols].mean(axis=1)
        lin_sinr = lin_sinr.replace(0, 1e-12)
        df["avg_sinr_db"] = 10 * np.log10(lin_sinr)
    else:
        df["avg_sinr_db"] = 0
        
    # Jamming Success (Ratio of nodes jammed)
    # Status 2 = Jammed
    status_cols = [c for c in df.columns if "node_" in c and "_status" in c]
    if status_cols:
        # Count 2s
        jammed_counts = (df[status_cols] == 2).sum(axis=1)
        df["jam_success_rate"] = jammed_counts / len(status_cols)
    else:
        if "jammed_count" in df.columns:
            # Assume 10 nodes (hardcoded fallback)
            df["jam_success_rate"] = df["jammed_count"] / 10.0
        else:
            df["jam_success_rate"] = 0
            
    return df

def main():
    sns.set_style("whitegrid")
    
    experiments = ["baseline", "ppo", "dqn"]
    labels = ["Baseline (QJC)", "Deep RL (PPO)", "Deep RL (DQN)"]
    colors = ["gray", "tab:blue", "tab:orange"]
    
    data_map = {}
    
    for exp, label, color in zip(experiments, labels, colors):
        df = load_data(exp)
        if df is not None:
            df = preprocess_df(df)
            data_map[label] = {"df": df, "color": color}
            
    if not data_map:
        print("No data found in experiments/ folder. Run experiments first.")
        return

    # Create Comparison Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 1. Jamming Success Rate
    ax1 = axes[0]
    for label, data in data_map.items():
        df = data["df"]
        # Smooth curve
        smoothed = df["jam_success_rate"].rolling(window=10, min_periods=1).mean()
        ax1.plot(df["step"], smoothed, label=label, color=data["color"], linewidth=2)
        
    ax1.set_title("Jamming Success Rate (Higher is Better for Attacker)", fontsize=14)
    ax1.set_ylabel("Success Ratio (0-1)", fontsize=12)
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    # 2. Impact on Network SINR
    ax2 = axes[1]
    for label, data in data_map.items():
        df = data["df"]
        smoothed = df["avg_sinr_db"].rolling(window=10, min_periods=1).mean()
        ax2.plot(df["step"], smoothed, label=label, color=data["color"], linewidth=2)
        
    ax2.set_title("Network Signal Quality (Lower is Better for Attacker)", fontsize=14)
    ax2.set_ylabel("Avg SINR (dB)", fontsize=12)
    ax2.legend()
    
    # 3. Energy Efficiency (Reward Proxy or Jammer Cost?)
    # Let's show Channel Matching - Hard to show 3 lines.
    # Let's show "Tracking Accuracy" -> Fraction of time on same channel as UAV
    ax3 = axes[2]
    
    for label, data in data_map.items():
        df = data["df"]
        if "uav_channel" in df.columns and "jammer_channel" in df.columns:
            match = (df["uav_channel"] == df["jammer_channel"]).astype(int)
            # Cumulative Average Match Rate
            cum_match = match.expanding().mean()
            ax3.plot(df["step"], cum_match, label=label, color=data["color"], linewidth=2)
            
    ax3.set_title("Channel Tracking Accuracy (Lock-on Capability)", fontsize=14)
    ax3.set_ylabel("Match Rate (Cumulative)", fontsize=12)
    ax3.set_xlabel("Simulation Step", fontsize=12)
    ax3.legend()
    ax3.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = "comparison_result.png"
    plt.savefig(save_path, dpi=300)
    print(f"Comparison Plot Saved: {save_path}")
    
    # Show
    print("Opening Plot...")
    plt.show()

if __name__ == "__main__":
    main()
