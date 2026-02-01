import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from core.config import UAVConfig
from core.env_config import EnvConfig



class Visualization:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.area_size = EnvConfig.AREA_SIZE
        plt.ion() # Interactive mode

    def render(self, env):
        self.ax.clear()
        
        # Area bounds
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        
        # UAV
        self.ax.scatter(env.uav.x, env.uav.y, c='blue', marker='^', s=100, label='UAV')
        
        # Nodes
        for node in env.nodes:
            if node.connection_status == 0: # Connected
                color = 'green'
                self.ax.plot([env.uav.x, node.x], [env.uav.y, node.y], color='green', linestyle='--', linewidth=0.5, alpha=0.5)
            elif node.connection_status == 1: # Out of Range
                color = 'gray'
            else: # Jammed (2)
                color = 'red'
                
            self.ax.scatter(node.x, node.y, c=color, marker='o', s=50, label='IoT Node' if node.id == 0 else "")
        
        # Attacker
        self.ax.scatter(env.attacker.x, env.attacker.y, c='red', marker='x', s=80, label='Attacker')
        
        # 3. Jamming Area Visualization (Contour)
        # Calculate SINR field for "What if a node was here?"
        if env.attacker.jamming_power > 1e-6:
            try:
                # Resolution for contour
                res = 50 
                xs = np.linspace(0, self.area_size, res)
                ys = np.linspace(0, self.area_size, res)
                X, Y = np.meshgrid(xs, ys)
                
                # Positions (Vectorized)
                # Grid Points (res x res)
                # UAV Pos
                uav_pos = np.array([env.uav.x, env.uav.y])
                jam_pos = np.array([env.attacker.x, env.attacker.y])
                
                # Distances
                # Dist to UAV (for Signal)
                d_uav = np.sqrt((X - uav_pos[0])**2 + (Y - uav_pos[1])**2)
                
                # Dist to Jammer (for Interference)
                d_jam = np.sqrt((X - jam_pos[0])**2 + (Y - jam_pos[1])**2)
                
                # Physics Calculations (Vectorized)
                import core.physics as physics
                from core.env_config import EnvConfig
                from core.config import UAVConfig

                # Beta
                beta_uav = physics.calculate_path_loss(d_uav)
                beta_jam = physics.calculate_path_loss(d_jam)
                
                # Powers
                # Signal: Node(at grid) -> UAV. P_Tx = EnvConfig.P_TX_NODE
                # Note: d_uav is correct distance.
                p_sig = physics.calculate_received_power(EnvConfig.P_TX_NODE, d_uav, beta_uav)
                
                # Jammer: Jammer -> Node(at grid). 
                # Note: Physics used dist(Node, Jammer). So d_jam is correct.
                p_jam = physics.calculate_received_power(env.attacker.jamming_power, d_jam, beta_jam)
                
                # SINR (Total with Jamming)
                sinr_total = physics.calculate_sinr(p_sig, UAVConfig.N0_Linear, p_jam)
                
                # SINR (No Jamming - Theoretical)
                sinr_no_jam = physics.calculate_sinr(p_sig, UAVConfig.N0_Linear, 0.0)
                
                # Visualization Logic:
                # We want to show RED only where:
                # 1. Connection IS broken (sinr_total < 1.0)
                # 2. Connection WOULD BE fine without jammer (sinr_no_jam > 1.0)
                # If sinr_no_jam < 1.0, it's Out of Range (Gray zone concept), not Jamming.
                
                # Create a field for contouring
                # We set values where sinr_no_jam < 1.0 to a high value (e.g. 10.0) 
                # so they don't fall into the < 1.0 contour level.
                visual_field = sinr_total.copy()
                visual_field[sinr_no_jam < 1.0] = 100.0 
                
                # Threshold (Linear 1.0)
                # Draw Contour
                # Levels: [0, 1.0] -> Jammed Region
                self.ax.contourf(X, Y, visual_field, levels=[0, 1.0], colors=['red'], alpha=0.2)
                self.ax.contour(X, Y, visual_field, levels=[1.0], colors=['red'], linestyles='--', linewidths=0.5)
                
                
            except Exception as e:
                print(f"Viz Error: {e}")

        # Info Screen
        info_text = f"Step: {env.current_step}\n"
        info_text += f"Jammer Power: {env.attacker.jamming_power:.2f} W\n"
        avg_sinr = sum([10*1.0 for n in env.nodes]) / len(env.nodes) # Placeholder
        # Real SINR could be retrieved from environment but visual is enough for now.
        
        self.ax.legend(loc='upper right')
        self.ax.set_title("IoT Based UAV Application - Security Simulation")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.draw()
        plt.pause(0.01) # Wait for UI update
