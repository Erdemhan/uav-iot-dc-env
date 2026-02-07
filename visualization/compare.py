
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

    # Print Statistical Summary
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON - STATISTICAL SUMMARY")
    print("="*70 + "\n")
    
    for label, data in data_map.items():
        df = data["df"]
        print(f"{'─'*70}")
        print(f"{label}")
        print(f"{'─'*70}")
        
        # Jamming Performance
        if "jammed_count" in df.columns:
            print(f"  Jamming Performance:")
            print(f"    • Avg Jammed Nodes: {df['jammed_count'].mean():.2f}")
            print(f"    • Total Jammed: {df['jammed_count'].sum()}")
            print(f"    • Max Jammed (single step): {df['jammed_count'].max()}")
            print(f"    • Success Rate: {df['jam_success_rate'].mean()*100:.1f}%")
        
        # Power Usage
        if "jammer_power" in df.columns:
            print(f"  Power Usage:")
            print(f"    • Avg Power: {df['jammer_power'].mean():.4f} W")
            print(f"    • Max Power: {df['jammer_power'].max():.4f} W")
            print(f"    • Power > 0 steps: {(df['jammer_power'] > 0).sum()}/100")
        
        # Channel Tracking
        if "uav_channel" in df.columns and "jammer_channel" in df.columns:
            match = (df["uav_channel"] == df["jammer_channel"]).astype(int)
            print(f"  Channel Tracking:")
            print(f"    • Match Rate: {match.mean()*100:.1f}%")
            print(f"    • Matched Steps: {match.sum()}/100")
        
        # Network Impact
        if "avg_sinr_db" in df.columns:
            print(f"  Network Impact:")
            print(f"    • Avg SINR: {df['avg_sinr_db'].mean():.2f} dB")
            print(f"    • Min SINR: {df['avg_sinr_db'].min():.2f} dB")
        
        print()
    
    print("="*70 + "\n")
    print("Generating comparison plots...\n")

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

    # Add summary statistics table to figure
    fig.text(0.5, 0.01, "Performance Summary", ha='center', fontsize=12, fontweight='bold')
    
    # Create summary table data
    table_data = []
    for label, data in data_map.items():
        df = data["df"]
        row = [
            label.replace("Deep RL ", ""),  # Shortened label
            f"{df['jammed_count'].mean():.2f}" if "jammed_count" in df.columns else "N/A",
            f"{df['jammer_power'].mean():.4f}" if "jammer_power" in df.columns else "N/A",
            f"{((df['uav_channel']==df['jammer_channel']).mean()*100):.1f}%" if "uav_channel" in df.columns else "N/A"
        ]
        table_data.append(row)
    
    # Add table at bottom
    table_ax = fig.add_axes([0.15, 0.02, 0.7, 0.08])  # [left, bottom, width, height]
    table_ax.axis('off')
    
    table = table_ax.table(
        cellText=table_data,
        colLabels=['Algorithm', 'Avg Jammed', 'Avg Power (W)', 'Channel Match'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    colors_map = {'Baseline (QJC)': 'gray', '(PPO)': 'tab:blue', '(DQN)': 'tab:orange'}
    for i, row in enumerate(table_data):
        for color_key, color in colors_map.items():
            if color_key in row[0]:
                table[(i+1, 0)].set_facecolor(color)
                table[(i+1, 0)].set_text_props(color='white', weight='bold')

    plt.subplots_adjust(bottom=0.12)  # Make room for table
    
    # Save statistics to CSV
    stats_path = "experiments/comparison_statistics.csv"
    with open(stats_path, 'w') as f:
        f.write("Algorithm,Avg_Jammed_Nodes,Total_Jammed,Max_Jammed,Success_Rate,Avg_Power_W,Max_Power_W,Channel_Match_Rate,Avg_SINR_dB\n")
        for label, data in data_map.items():
            df = data["df"]
            f.write(f"{label},")
            f.write(f"{df['jammed_count'].mean():.2f}," if "jammed_count" in df.columns else "N/A,")
            f.write(f"{df['jammed_count'].sum()}," if "jammed_count" in df.columns else "N/A,")
            f.write(f"{df['jammed_count'].max()}," if "jammed_count" in df.columns else "N/A,")
            f.write(f"{df['jam_success_rate'].mean()*100:.1f}," if "jam_success_rate" in df.columns else "N/A,")
            f.write(f"{df['jammer_power'].mean():.4f}," if "jammer_power" in df.columns else "N/A,")
            f.write(f"{df['jammer_power'].max():.4f}," if "jammer_power" in df.columns else "N/A,")
            if "uav_channel" in df.columns and "jammer_channel" in df.columns:
                match_rate = (df["uav_channel"] == df["jammer_channel"]).mean() * 100
                f.write(f"{match_rate:.1f},")
            else:
                f.write("N/A,")
            f.write(f"{df['avg_sinr_db'].mean():.2f}\n" if "avg_sinr_db" in df.columns else "N/A\n")
    
    print(f"Statistics saved to: {stats_path}")
    
    save_path = "experiments/comparison_result.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison Plot Saved: {save_path}")
    
    # Show
    print("Opening Plot...")
    plt.show()

if __name__ == "__main__":
    main()
