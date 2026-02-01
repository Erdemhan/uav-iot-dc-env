import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from core.env_config import EnvConfig


class SimulationVisualizer:
    def __init__(self, exp_dir: str = None):
        if exp_dir is None:
            exp_dir = self.find_latest_experiment()
            
        self.exp_dir = exp_dir
        self.csv_path = os.path.join(exp_dir, "history.csv")
        self.config_path = os.path.join(exp_dir, "config.json")
        
        # Load Data
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Data Loaded: {len(self.df)} steps.")
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV Not Found: {self.csv_path}")

        # Load Config (for static positions if needed)
        self.config = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = json.load(f)

        # Preprocess Data
        self._preprocess_data()
        
        # Style
        sns.set_style("whitegrid")

    def find_latest_experiment(self, logs_dir="logs") -> str:
        """Finds the latest experiment folder."""
        all_dirs = glob.glob(os.path.join(logs_dir, "EXP_*"))
        if not all_dirs:
            raise FileNotFoundError("No experiment logs found (logs/ empty).")
        latest_dir = max(all_dirs, key=os.path.getmtime)
        return latest_dir

    def _preprocess_data(self):
        """Prepares raw data for analysis."""
        # 1. Calculate Average System SINR (dB)
        # Find all columns starting with 'node_' and ending with '_sinr'
        sinr_cols = [c for c in self.df.columns if "node_" in c and "_sinr" in c]
        if sinr_cols:
            # Linear avg
            avg_linear_sinr = self.df[sinr_cols].mean(axis=1)
            # Avoid log(0) error
            avg_linear_sinr = avg_linear_sinr.replace(0, 1e-12) 
            self.df["sinr_db"] = 10 * np.log10(avg_linear_sinr)
        else:
            self.df["sinr_db"] = 0.0

        # 2. Calculate Average AoI
        aoi_cols = [c for c in self.df.columns if "node_" in c and "_aoi" in c]
        if aoi_cols:
            self.df["aoi_avg"] = self.df[aoi_cols].mean(axis=1)
        else:
            self.df["aoi_avg"] = 0.0

        # 3. Status Flags
        # Check node status columns
        status_cols = [c for c in self.df.columns if "node_" in c and "_status" in c]
        
        if status_cols:
            # Axis 1 max: if any node is 2, result is 2. If max is 1, result 1.
            self.df["step_status"] = self.df[status_cols].max(axis=1)
        else:
            # Fallback if logs old
            self.df["step_status"] = 0
            if "jammed_count" in self.df.columns:
                 self.df.loc[self.df["jammed_count"] > 0, "step_status"] = 2

    def plot_trajectory(self):
        """Draws UAV and Attacker trajectory."""
        plt.figure(figsize=(10, 10))
        
        # 1. IoT Nodes (Green Squares)
        # Get position of first step for each node (Assuming static)
        # Find node_x columns via regex
        node_x_cols = sorted([c for c in self.df.columns if "node_" in c and "_x" in c])
        node_y_cols = sorted([c for c in self.df.columns if "node_" in c and "_y" in c])
        
        if node_x_cols and node_y_cols:
            # Take only the first row (Step 0)
            xs = self.df.iloc[0][node_x_cols].values
            ys = self.df.iloc[0][node_y_cols].values
            plt.scatter(xs, ys, color="green", marker="s", s=100, label="IoT Nodes", zorder=4)
            
            # Write Node IDs
            for i, (Nx, Ny) in enumerate(zip(xs, ys)):
                plt.text(Nx+10, Ny+10, f"N{i}", fontsize=9, color="green")

        # 2. UAV Path (Line)
        plt.plot(self.df["uav_x"], self.df["uav_y"], color="gray", alpha=0.5, linewidth=2, label="UAV Path")
        
        # 3. UAV States (Scatter on Path)
        # Status 0: Connected / Safe -> Green
        safe_points = self.df[self.df["step_status"] == 0]
        if not safe_points.empty:
            plt.scatter(safe_points["uav_x"], safe_points["uav_y"], 
                        color="lime", s=30, alpha=0.6, label="Safe / Connected")
            
        # Status 1: Out of Range -> Gray
        range_points = self.df[self.df["step_status"] == 1]
        if not range_points.empty:
            plt.scatter(range_points["uav_x"], range_points["uav_y"], 
                        color="gray", s=40, marker="o", alpha=0.5, label="Out of Range")

        # Status 2: Jammed -> Red
        jammed_points = self.df[self.df["step_status"] == 2]
        if not jammed_points.empty:
            plt.scatter(jammed_points["uav_x"], jammed_points["uav_y"], 
                        color="red", s=50, marker="x", label="Jamming Detected")

        # 4. Start/End Points
        plt.scatter(self.df["uav_x"].iloc[0], self.df["uav_y"].iloc[0], color="blue", marker="^", s=150, label="Start", zorder=5)
        plt.scatter(self.df["uav_x"].iloc[-1], self.df["uav_y"].iloc[-1], color="blue", marker="s", s=100, label="End", zorder=5)

        # 5. Attacker Position
        area_size = float(self.config.get("AREA_SIZE", EnvConfig.AREA_SIZE))
        att_x = float(self.config.get("ATTACKER_POS_X", EnvConfig.ATTACKER_POS_X))
        att_y = float(self.config.get("ATTACKER_POS_Y", EnvConfig.ATTACKER_POS_Y))
        plt.scatter(att_x, att_y, color="darkred", marker="X", s=250, label="Attacker", zorder=6)
        
        # Draw Jamming Effective Zone (approx) just for visual aid? 
        # No, dynamic power makes it hard.
        
        plt.title("UAV Trajectory & Network Status", fontsize=14)
        plt.xlabel("Position X (m)")
        plt.ylabel("Position Y (m)")
        plt.legend(loc='upper right', frameon=True)
        plt.xlim(0, area_size)
        plt.ylim(0, area_size)
        
        save_path = os.path.join(self.exp_dir, "trajectory.png")
        plt.savefig(save_path, dpi=300)
        print(f"Trajectory saved: {save_path}")
        plt.close()

    def plot_metrics(self):
        """Draws time series metrics (SINR, AoI, Energy)."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        steps = self.df["step"]

        # Subplot 1: SINR (dB) vs Jamming Power
        ax1 = axes[0]
        color = 'tab:blue'
        ax1.set_ylabel('Avg SINR (dB)', color=color, fontsize=12)
        ax1.plot(steps, self.df["sinr_db"], color=color, linewidth=2, label="SINR (dB)")
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Threshold Line (Example: 0 dB)
        ax1.axhline(0, color='gray', linestyle='--', label="Threshold (0 dB)")

        # Twin Axis for Jamming
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Jamming Power (W)', color=color, fontsize=12)
        ax2.plot(steps, self.df["jammer_power"], color=color, linestyle=':', linewidth=2, label="Jamming Power")
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax1.set_title("Communication Quality vs Jamming Attack", fontsize=14)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Subplot 2: Age of Information (AoI)
        ax_aoi = axes[1]
        ax_aoi.plot(steps, self.df["aoi_avg"], color="green", label="Avg AoI")
        ax_aoi.set_ylabel("Avg AoI (s)", fontsize=12)
        ax_aoi.set_title("Information Freshness (Age of Information)", fontsize=14)
        ax_aoi.legend()
        ax_aoi.grid(True, which="both", linestyle='--')

        # Subplot 3: UAV Energy Consumption
        ax_eng = axes[2]
        ax_eng.plot(steps, self.df["uav_energy"], color="orange", linewidth=2, label="Total Energy")
        ax_eng.set_xlabel("Time Step", fontsize=12)
        ax_eng.set_ylabel("Energy (J)", fontsize=12)
        ax_eng.set_title("UAV Total Energy Consumption", fontsize=14)
        ax_eng.legend()

        plt.tight_layout()
        save_path = os.path.join(self.exp_dir, "metrics_analysis.png")
        plt.savefig(save_path, dpi=300)
        print(f"Metrics analysis saved: {save_path}")
        plt.close()

    def generate_report(self):
        """Triggers report generation."""
        print(f"--- Visualization Report: {self.exp_dir} ---")
        self.plot_trajectory()
        self.plot_metrics()
        print("Visualization Completed.")

if __name__ == "__main__":
    viz = SimulationVisualizer()
    viz.generate_report()
