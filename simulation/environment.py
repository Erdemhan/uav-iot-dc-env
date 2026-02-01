import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math


from core.config import UAVConfig
from core.env_config import EnvConfig
from .entities import UAVAgent, IoTNode, SmartAttacker
import core.physics as physics
from core.logger import SimulationLogger


class UAV_IoT_Env(gym.Env):
    """
    OpenAI Gymnasium compatible UAV-IoT Simulation Environment.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, logger: SimulationLogger = None):
        super().__init__()
        
        self.logger = logger
        
        # Area bounds
        self.area_size = EnvConfig.AREA_SIZE
        
        # Create Agents
        self.uav = UAVAgent(x=EnvConfig.UAV_START_X, y=EnvConfig.UAV_START_Y, z=UAVConfig.H)
        
        self.nodes = []
        for i in range(EnvConfig.NUM_NODES):
            # Place randomly
            nx = np.random.uniform(0, self.area_size)
            ny = np.random.uniform(0, self.area_size)
            self.nodes.append(IoTNode(i, nx, ny))
            
        # Fixed position from EnvConfig
        # Note: EnvConfig has ATTACKER_POS_X/Y. If we want relative to center + 100 as before:
        # self.attacker = SmartAttacker(x=self.area_size/2 + 100, ... )
        # But we added ATTACKER_POS_X/Y to config, let's use them directly or if they are offsets.
        # User asked to separate parameters. Let's use absolute positions from EnvConfig if defined roughly match.
        # EnvConfig defined 600, 600. (Center is 500,500). So it matches.
        self.attacker = SmartAttacker(x=EnvConfig.ATTACKER_POS_X, y=EnvConfig.ATTACKER_POS_Y)
        
        # Action Space: Attacker Jamming Power (Continuous)
        self.action_space = spaces.Box(low=0.0, high=EnvConfig.MAX_JAMMING_POWER, shape=(1,), dtype=np.float32)
        
        # Observation Space:
        # UAV (x, y), Attacker (x, y), Nodes (x, y) * N, Node SINR * N, Node AoI * N
        # Total size: 2 + 2 + 2*N + N + N = 4 + 4*N
        obs_dim = 4 + 4 * EnvConfig.NUM_NODES
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = EnvConfig.MAX_STEPS # Steps per episode
        self.dt = EnvConfig.STEP_TIME # Time step (seconds)
        
        # Angle for circular motion
        self.uav_angle = np.radians(EnvConfig.UAV_START_ANGLE)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.uav_angle = 0.0
        
        # Centers UAV
        self.uav.x = self.area_size/2
        self.uav.y = self.area_size/2
        self.uav.total_energy_consumed = 0.0
        
        # Reset Nodes
        for node in self.nodes:
            node.aoi = 0.0
            node.total_energy_consumed = 0.0
            
        self.attacker.set_jamming_power(0.0)
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        obs.extend([self.uav.x, self.uav.y])
        obs.extend([self.attacker.x, self.attacker.y])
        
        sinrs = []
        aois = []
        for node in self.nodes:
            obs.extend([node.x, node.y])
            aois.append(node.aoi)
            # SINR should be calculated instantaneously, storing the last known value here could work 
            # but calculating it inside step and putting it into state is more accurate. 
            # Putting 0 placeholder for now, returns current in step function.
            sinrs.append(0.0) 
            
        obs.extend(sinrs)
        obs.extend(aois)
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        # 1. Apply Action (Attacker)
        jam_power = float(action[0])
        self.attacker.set_jamming_power(jam_power)
        
        # 2. UAV Movement (Simple Circular Trajectory)
        # Circle around center with r=200m
        radius = EnvConfig.UAV_RADIUS
        center_x, center_y = self.area_size/2, self.area_size/2
        speed = EnvConfig.UAV_SPEED # m/s
        
        # Angular speed w = v / r
        angular_speed = speed / radius
        self.uav_angle += angular_speed * self.dt
        
        new_x = center_x + radius * np.cos(self.uav_angle)
        new_y = center_y + radius * np.sin(self.uav_angle)
        
        # Update velocity vector
        self.uav.vx = (new_x - self.uav.x) / self.dt
        self.uav.vy = (new_y - self.uav.y) / self.dt
        
        self.uav.x = new_x
        self.uav.y = new_y
        
        # UAV Energy Consumption
        v_uav = self.uav.velocity_magnitude
        self.uav.consume_energy(v_uav, self.dt)
        
        # 3. Physics Calculations (For each node)
        total_sinr = 0
        jammed_count = 0
        step_log = {
            "step": self.current_step,
            "uav_x": self.uav.x,
            "uav_y": self.uav.y,
            "jammer_power": jam_power,
            "uav_energy": self.uav.total_energy_consumed
        }
        
        for i, node in enumerate(self.nodes):
            # Distance
            dist_uav = np.linalg.norm(node.position - self.uav.position)
            dist_jam = np.linalg.norm(node.position - self.attacker.position)
            
            # Channel Gains
            beta_uav = physics.calculate_path_loss(dist_uav)
            # If jammer is on ground level, from jammer to node:
            # Using same path loss model for simplicity (or can be different)
            beta_jam = physics.calculate_path_loss(dist_jam)
            
            # Powers
            p_rx_signal = physics.calculate_received_power(node.tx_power, dist_uav, beta_uav)
            p_rx_jam = physics.calculate_received_power(self.attacker.jamming_power, dist_jam, beta_jam)
            
            # SINR (With Jamming)
            sinr = physics.calculate_sinr(p_rx_signal, UAVConfig.N0_Linear, p_rx_jam)
            total_sinr += sinr
            
            # SINR (No Jamming) - To check if it is Out of Range
            sinr_no_jam = physics.calculate_sinr(p_rx_signal, UAVConfig.N0_Linear, 0.0)
            
            # Rate
            rate = physics.calculate_data_rate(UAVConfig.B, sinr)
            
            # Connection Status Logic
            # Threshold SINR > 1.0 (0 dB)
            connected_now = sinr > 1.0
            connected_theoretical = sinr_no_jam > 1.0
            
            status = 0 # Connected
            if connected_now:
                status = 0
            else:
                if not connected_theoretical:
                    status = 1 # Out of Range (Even without jammer, it would fail)
                else:
                    status = 2 # Jammed (Ideally connected, but jammer broke it)
                    jammed_count += 1
            
            # Update Node Status
            node.connection_status = status
            node.update_aoi(self.dt, success=(status == 0))
            node.consume_energy(rate)
            
            # Log Individual Node Metrics
            step_log[f"node_{i}_x"] = node.x
            step_log[f"node_{i}_y"] = node.y
            step_log[f"node_{i}_sinr"] = sinr
            step_log[f"node_{i}_aoi"] = node.aoi
            step_log[f"node_{i}_energy"] = node.total_energy_consumed
            step_log[f"node_{i}_status"] = status

        # 4. Reward: Attacker's goal is to cut communication (+1 per jammed node)
        reward = jammed_count
        
        # 5. Logging
        step_log["reward"] = reward
        step_log["jammed_count"] = jammed_count
        
        if self.logger:
            self.logger.log_step(step_log)
            
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = step_log
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        # Render will be done by visualization.py, returning state might be enough
        # or Visualization class can be passed here as parameter.
        # Visualization.render(env) call will be made in Main file.
        pass
