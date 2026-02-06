import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from confs.env_config import EnvConfig


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
            # Custom aggregation: If ANY node is Connected (0), status is 0 (to suppress generic Jammed/Range markers).
            # If no connection, but ANY Jammed (2), status is 2.
            # Else (all 1s), status is 1.
            
            # 1. Default to Out of Range (1)
            self.df["step_status"] = 1
            
            # 2. Check for Jammed (2)
            is_jammed = (self.df[status_cols] == 2).any(axis=1)
            self.df.loc[is_jammed, "step_status"] = 2
            
            # 3. Check for Connected (0) - This overwrites Jammed!
            is_connected = (self.df[status_cols] == 0).any(axis=1)
            self.df.loc[is_connected, "step_status"] = 0
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
            # Take only the first row (Step 0)
            xs = self.df.iloc[0][node_x_cols].values
            ys = self.df.iloc[0][node_y_cols].values
            
            # Plot each node with its distinct color
            cmap = plt.get_cmap('tab10')
            node_colors = [cmap(i) for i in range(10)]
            
            for i, (Nx, Ny) in enumerate(zip(xs, ys)):
                # Use modulo for color index (safe for >10 nodes)
                c = node_colors[i % 10]
                plt.scatter(Nx, Ny, color=c, marker="s", s=100, label="IoT Nodes" if i==0 else "", zorder=4, edgecolors='black')
                plt.text(Nx+10, Ny+10, f"N{i}", fontsize=9, color="black", fontweight='bold')

        # 2. UAV Path (Line)
        plt.plot(self.df["uav_x"], self.df["uav_y"], color="gray", alpha=0.5, linewidth=2, label="UAV Path")
        
        # 3. UAV States (Scatter on Path)
        # Status 0: Connected -> Distinct Colors with Offset
        cmap = plt.get_cmap('tab10')
        node_colors = [cmap(i) for i in range(10)]
        
        # Identify node status columns
        node_status_cols = [c for c in self.df.columns if "node_" in c and "_status" in c]
        num_nodes = len(node_status_cols)
        
        for i, col in enumerate(node_status_cols):
             # Extract Node ID from col name (node_0_status -> 0)
             try:
                 node_id = int(col.split("_")[1])
             except:
                 node_id = i
                 
             # Filter connected steps for this node
             connected = self.df[self.df[col] == 0]
             
             if not connected.empty:
                 # Calculate Offset to prevent overlap
                 # Radial spread or linear shift. Linear is simpler.
                 # Shift range: [-30m, +30m]
                 if num_nodes > 1:
                     offset_mag = 15.0 
                     # angle = 2*pi * i / num_nodes
                     # dx = np.cos(angle) * offset_mag
                     # dy = np.sin(angle) * offset_mag
                     # Or simple linear shift
                     dx = (i - num_nodes/2.0) * 10.0
                     dy = (i % 2) * 10.0 # Stagger Y slightly
                 else:
                     dx, dy = 0, 0
                     
                 plt.scatter(connected["uav_x"] + dx, connected["uav_y"] + dy, 
                             color=node_colors[node_id % 10], s=25, alpha=0.7, 
                             label=f"N{node_id} Connected" if i < 5 else "") # Limit labels

            
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

    def plot_advanced_metrics(self):
        """Draws per-node communication statistics (Success Duration, Max Streak)."""
        # Find relevant columns
        total_time_cols = sorted([c for c in self.df.columns if "node_" in c and "_total_time" in c])
        max_cont_cols = sorted([c for c in self.df.columns if "node_" in c and "_max_continuous_time" in c])
        
        if not total_time_cols:
            print("Advanced metrics (total_time/max_continuous_time) not found in logs.")
            return

        # Get final values (last step)
        final_total = self.df.iloc[-1][total_time_cols].values
        final_max = self.df.iloc[-1][max_cont_cols].values
        
        num_nodes = len(total_time_cols)
        node_indices = np.arange(num_nodes)
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # 1. Total Successful Duration
        ax1 = axes[0]
        ax1.bar(node_indices, final_total, color='skyblue', edgecolor='black')
        ax1.set_title("Total Successful Communication Duration per Node", fontsize=14)
        ax1.set_ylabel("Duration (seconds)", fontsize=12)
        ax1.set_xticks(node_indices)
        ax1.set_xticklabels([f"Node {i}" for i in range(num_nodes)])
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top
        for i, v in enumerate(final_total):
            ax1.text(i, v + 0.1, f"{v:.1f}s", ha='center', va='bottom')
            
        # Global Avg Annotation
        avg_total = np.mean(final_total)
        ax1.axhline(avg_total, color='red', linestyle='--', linewidth=1.5, label=f"Avg: {avg_total:.2f}s")
        ax1.legend()

        # 2. Max Continuous Duration
        ax2 = axes[1]
        ax2.bar(node_indices, final_max, color='salmon', edgecolor='black')
        ax2.set_title("Max Continuous Connection Streak per Node", fontsize=14)
        ax2.set_ylabel("Duration (seconds)", fontsize=12)
        ax2.set_xticks(node_indices)
        ax2.set_xticklabels([f"Node {i}" for i in range(num_nodes)])
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(final_max):
            ax2.text(i, v + 0.1, f"{v:.1f}s", ha='center', va='bottom')

        plt.tight_layout()
        save_path = os.path.join(self.exp_dir, "advanced_metrics.png")
        plt.savefig(save_path, dpi=300)
        print(f"Advanced metrics saved: {save_path}")
        plt.close()

    def show_dashboard(self):
        """Displays all generated plots in a single window."""
        img_traj_path = os.path.join(self.exp_dir, "trajectory.png")
        img_metrics_path = os.path.join(self.exp_dir, "metrics_analysis.png")
        img_adv_path = os.path.join(self.exp_dir, "advanced_metrics.png")
        
        images = []
        titles = []
        
        if os.path.exists(img_traj_path):
            images.append(plt.imread(img_traj_path))
            titles.append("Trajectory Analysis")
        if os.path.exists(img_metrics_path):
            images.append(plt.imread(img_metrics_path))
            titles.append("Metrics Analysis")
        if os.path.exists(img_adv_path):
            images.append(plt.imread(img_adv_path))
            titles.append("Advanced Communication Stats")
            
        if not images:
            print("No images to display in dashboard.")
            return
            
        # Create a grid
        n_imgs = len(images)
        cols = 3 if n_imgs >= 3 else n_imgs
        # Adjust layout dynamically? For now just 1 row
        
        fig, axes = plt.subplots(1, n_imgs, figsize=(6 * n_imgs, 6))
        
        if n_imgs == 1:
            axes = [axes]
            
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=12)
            
        plt.tight_layout()
        plt.suptitle(f"Simulation Dashboard: {os.path.basename(self.exp_dir)}", fontsize=16)
        
        # Bring window to front (OS dependent, experimental)
        try:
            mng = plt.get_current_fig_manager()
            # TkAgg backend
            if hasattr(mng, 'window'):
               mng.window.attributes('-topmost', 1)
               mng.window.attributes('-topmost', 0)
        except:
            pass
            
        print("Opening Dashboard Window...")
        plt.ioff() # Disable interactive mode to ensure blocking
        plt.show(block=True)

    def generate_report(self):
        """Triggers report generation."""
        print(f"--- Visualization Report: {self.exp_dir} ---")
        self.plot_trajectory()
        self.plot_metrics()
        self.plot_advanced_metrics()
        print("Visualization Completed.")

if __name__ == "__main__":
    viz = SimulationVisualizer()
    viz.generate_report()
