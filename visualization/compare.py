import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def load_data_from_run_dir(run_dir, algo_name):
    """Load evaluation history.csv from run directory structure"""
    path = os.path.join(run_dir, algo_name, "evaluation", "history.csv")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory (e.g., artifacts/2026-02-08_22-51-00)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for comparison results (defaults to run-dir/comparison)")
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_dir, "comparison")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    experiments = ["baseline", "ppo", "dqn", "ppo_lstm"]
    labels = ["Baseline (QJC)", "Deep RL (PPO)", "Deep RL (DQN)", "Deep RL (PPO-LSTM)"]
    colors = ["gray", "tab:blue", "tab:orange", "tab:purple"]
    
    data_map = {}
    
    for exp, label, color in zip(experiments, labels, colors):
        df = load_data_from_run_dir(args.run_dir, exp)
        if df is not None:
            df = preprocess_df(df)
            data_map[label] = {"df": df, "color": color, "exp_name": exp}
            
    if not data_map:
        print(f"No data found in {args.run_dir}. Run experiments first.")
        return

    # Print Statistical Summary
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON - STATISTICAL SUMMARY")
    print("="*70 + "\n")
    
    for label, data in data_map.items():
        df = data["df"]
        print(f"{'-'*70}")
        print(f"{label}")
        print(f"{'-'*70}")
        print("  Jamming Performance:")
        if "jammed_count" in df.columns:
            print(f"    - Avg Jammed Nodes: {df['jammed_count'].mean():.2f}")
            print(f"    - Total Jammed: {df['jammed_count'].sum()}")
            print(f"    - Max Jammed (single step): {df['jammed_count'].max()}")
        print(f"    - Success Rate: {df['jam_success_rate'].mean()*100:.1f}%")
        
        print("  Power Usage:")
        if "jammer_power" in df.columns:
            print(f"    - Avg Power: {df['jammer_power'].mean():.4f} W")
            print(f"    - Max Power: {df['jammer_power'].max():.4f} W")
            print(f"    - Power > 0 steps: {(df['jammer_power'] > 0).sum()}/{len(df)}")
        
        print("  Channel Tracking:")
        if "uav_channel" in df.columns and "jammer_channel" in df.columns:
            match_rate = (df["uav_channel"] == df["jammer_channel"]).mean() * 100
            print(f"    - Channel Match Rate: {match_rate:.1f}%")
        
        print("  Network SINR:")
        print(f"    - Avg SINR: {df['avg_sinr_db'].mean():.2f} dB")
        print(f"    - Max SINR: {df['avg_sinr_db'].max():.2f} dB")
        print(f"    - Min SINR: {df['avg_sinr_db'].min():.2f} dB")
        print()
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot 1: Jamming Success Rate
    ax1 = axes[0]
    for label, data in data_map.items():
        df = data["df"]
        color = data["color"]
        ax1.plot(df.index, df["jam_success_rate"] * 100, label=label, color=color, linewidth=2, alpha=0.8)
    ax1.set_title("Jamming Success Rate Over Time", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Success Rate (%)", fontsize=12)
    ax1.set_xlabel("Time Step", fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Network SINR
    ax2 = axes[1]
    for label, data in data_map.items():
        df = data["df"]
        color = data["color"]
        ax2.plot(df.index, df["avg_sinr_db"], label=label, color=color, linewidth=2, alpha=0.8)
    ax2.set_title("Network SINR (Lower = Better Jamming)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("SINR (dB)", fontsize=12)
    ax2.set_xlabel("Time Step", fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Channel Tracking Accuracy
    ax3 = axes[2]
    for label, data in data_map.items():
        df = data["df"]
        color = data["color"]
        if "uav_channel" in df.columns and "jammer_channel" in df.columns:
            # Rolling average for smoothness
            match = (df["uav_channel"] == df["jammer_channel"]).rolling(5, min_periods=1).mean() * 100
            ax3.plot(df.index, match, label=label, color=color, linewidth=2, alpha=0.8)
    ax3.set_title("Channel Tracking Accuracy (5-step Rolling Avg)", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Match Rate (%)", fontsize=12)
    ax3.set_xlabel("Time Step", fontsize=12)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training Curves
    ax4 = axes[3]
    
    # Load training data
    # Load Config to get TRAIN_BATCH_SIZE
    from confs.model_config import GlobalConfig
    BATCH_SIZE = GlobalConfig.TRAIN_BATCH_SIZE

    # Baseline
    baseline_train_path = os.path.join(args.run_dir, "baseline", "training_curve.csv")
    if os.path.exists(baseline_train_path):
        from confs.env_config import EnvConfig
        df_train = pd.read_csv(baseline_train_path)
        # Calculate total_steps: episode * MAX_STEPS (100)
        df_train['total_steps'] = df_train['episode'] * EnvConfig.MAX_STEPS
        
        # Resample to align with Deep RL (average over BATCH_SIZE steps)
        # We want steps 1-1000 to be plotted at X=1000.
        df_train['step_bin'] = ((df_train['total_steps'] - 1) // BATCH_SIZE) * BATCH_SIZE + BATCH_SIZE
        
        # Group by bin and take mean of 'total_reward'
        df_resampled = df_train.groupby('step_bin')['total_reward'].mean().reset_index()
        
        ax4.plot(df_resampled['step_bin'], df_resampled['total_reward'], 
                label='Baseline (QJC)', color='gray', linewidth=2, alpha=0.8, marker='o', markersize=4)

    # PPO
    ppo_progress_files = glob.glob(os.path.join(args.run_dir, "ppo", "PPO_*/*/progress.csv"))
    if ppo_progress_files:
        latest_ppo_file = max(ppo_progress_files, key=os.path.getmtime)
        if os.path.exists(latest_ppo_file):
            df_ppo = pd.read_csv(latest_ppo_file)
            # Use Ray 2.x column names
            if 'timesteps_total' in df_ppo.columns and 'env_runners/episode_reward_mean' in df_ppo.columns:
                ax4.plot(df_ppo['timesteps_total'], df_ppo['env_runners/episode_reward_mean'], 
                        label='PPO', color='tab:blue', linewidth=2, alpha=0.8, marker='o', markersize=4)
    
    # DQN
    dqn_progress_files = glob.glob(os.path.join(args.run_dir, "dqn", "DQN_*/*/progress.csv"))
    if dqn_progress_files:
        latest_dqn_file = max(dqn_progress_files, key=os.path.getmtime)
        if os.path.exists(latest_dqn_file):
            df_dqn = pd.read_csv(latest_dqn_file)
            # Use Ray 2.x column names
            if 'timesteps_total' in df_dqn.columns and 'env_runners/episode_reward_mean' in df_dqn.columns:
                ax4.plot(df_dqn['timesteps_total'], df_dqn['env_runners/episode_reward_mean'], 
                        label='DQN', color='tab:orange', linewidth=2, alpha=0.8, marker='o', markersize=4)

    # PPO-LSTM
    ppo_lstm_progress_files = glob.glob(os.path.join(args.run_dir, "ppo_lstm", "PPO_*/*/progress.csv"))
    if ppo_lstm_progress_files:
        latest_ppo_lstm_file = max(ppo_lstm_progress_files, key=os.path.getmtime)
        if os.path.exists(latest_ppo_lstm_file):
            df_ppo_lstm = pd.read_csv(latest_ppo_lstm_file)
            if 'timesteps_total' in df_ppo_lstm.columns and 'env_runners/episode_reward_mean' in df_ppo_lstm.columns:
                ax4.plot(df_ppo_lstm['timesteps_total'], df_ppo_lstm['env_runners/episode_reward_mean'], 
                        label='PPO-LSTM', color='tab:purple', linewidth=2, alpha=0.8, marker='o', markersize=4)
    
    ax4.set_title("Training Progress: Reward Learning Curves", fontsize=14, fontweight='bold')
    ax4.set_ylabel("Mean Episode Reward", fontsize=12)
    ax4.set_xlabel("Total Environment Steps", fontsize=12)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.suptitle("UAV-IoT Jamming: Algorithm Comparison", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
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
    colors_map = {'Baseline (QJC)': 'gray', '(PPO)': 'tab:blue', '(DQN)': 'tab:orange', '(PPO-LSTM)': 'tab:purple'}
    for i, row in enumerate(table_data):
        for color_key, color in colors_map.items():
            if color_key in row[0]:
                table[(i+1, 0)].set_facecolor(color)
                table[(i+1, 0)].set_text_props(color='white', weight='bold')

    plt.subplots_adjust(bottom=0.12)  # Make room for table
    
    # Save statistics to CSV
    stats_path = os.path.join(args.output_dir, "comparison_statistics.csv")
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
    
    save_path = os.path.join(args.output_dir, "comparison_result.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison Plot Saved: {save_path}")
    
    # Show
    print("Opening Plot...")
    plt.show()

if __name__ == "__main__":
    main()
