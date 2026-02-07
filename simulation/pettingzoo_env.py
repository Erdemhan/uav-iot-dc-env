from pettingzoo.utils import ParallelEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from confs.config import UAVConfig
from confs.env_config import EnvConfig
from simulation.entities import UAVAgent, IoTNode, SmartAttacker
from simulation.controllers import UAVRuleBasedController
import core.physics as physics
from core.logger import SimulationLogger

class UAV_IoT_PZ_Env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "uav_iot_v1"}

    def __init__(self, logger: SimulationLogger = None, auto_uav: bool = False, flatten_actions: bool = False):
        super().__init__()
        self.logger = logger
        self.area_size = EnvConfig.AREA_SIZE
        self.auto_uav = auto_uav
        self.flatten_actions = flatten_actions
        
        # --- Agents Configuration ---
        # If auto_uav is True, UAV is part of environment (not an agent)
        if self.auto_uav:
            self.possible_agents = ["jammer_0"] + [f"node_{i}" for i in range(EnvConfig.NUM_NODES)]
        else:
            self.possible_agents = ["uav_0", "jammer_0"] + [f"node_{i}" for i in range(EnvConfig.NUM_NODES)]
            
        self.agents = self.possible_agents[:]
        
        # --- Entities ---
        # These will be reset in self.reset()
        self.uav = None
        self.attacker = None
        self.nodes = []
        
        # Internal Controller for Auto Mode
        self.uav_controller = None

        # --- Spaces ---
        self.observation_spaces = {}
        self.action_spaces = {}
        
        # Common Observation Space Dimension
        # Sensing Mode:
        # Instead of UAV(x,y) and Jammer(x,y) -> Just Distance (1)
        # Channels (2)
        # Nodes (x,y normalized) * N (Nodes are static/known map) or also Distance?
        # Let's keep Nodes as map knowledge (assuming Jammer knows where IoT devices are fixed).
        # obs_dim = 1 + 2 + 3 * N  (Dist_UAV + Chs + Node(x,y,AoI))
        obs_dim = 1 + 2 + 3 * EnvConfig.NUM_NODES
        common_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        for agent in self.possible_agents:
            self.observation_spaces[agent] = common_obs_space
            
            if agent == "uav_0":
                # Action: Velocity Vector [vx, vy]
                # We assume a reasonable max speed bound for the action space definition, 
                # though physics limits it strictly.
                max_speed = EnvConfig.UAV_SPEED * 2 
                self.action_spaces[agent] = spaces.Box(low=-max_speed, high=max_speed, shape=(2,), dtype=np.float32)
            
            elif agent == "jammer_0":
                # Action: [Channel (3), Power Level (10)]
                if self.flatten_actions:
                    self.action_spaces[agent] = spaces.Discrete(30)
                else:
                    self.action_spaces[agent] = spaces.MultiDiscrete([3, 10])
            
            else: # Nodes
                # Action: No-Op (Discrete(1))
                self.action_spaces[agent] = spaces.Discrete(1)

        self.current_step = 0
        self.max_steps = EnvConfig.MAX_STEPS
        self.dt = EnvConfig.STEP_TIME

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        
        # Reset Entities
        self.uav = UAVAgent(x=EnvConfig.UAV_START_X, y=EnvConfig.UAV_START_Y, z=UAVConfig.H)
        self.uav.x = self.area_size / 2 # Center start as per previous logic logic
        self.uav.y = self.area_size / 2
        
        self.attacker = SmartAttacker(x=EnvConfig.ATTACKER_POS_X, y=EnvConfig.ATTACKER_POS_Y)
        
        self.nodes = []
        # Re-seed numpy for node placement if seed provided
        if seed is not None:
            np.random.seed(seed)
            
        for i in range(EnvConfig.NUM_NODES):
            nx = np.random.uniform(0, self.area_size)
            ny = np.random.uniform(0, self.area_size)
            self.nodes.append(IoTNode(i, nx, ny))

        if self.auto_uav:
            self.uav_controller = UAVRuleBasedController(self)
            self.uav_controller.reset()

        observations = {agent: self._get_obs() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        self.current_step += 1
        
        # 1. Apply Actions
        # UAV
        uav_action = None
        
        if self.auto_uav:
            # Get Action from Internal Controller
            uav_action = self.uav_controller.get_action()
        elif "uav_0" in actions:
            uav_action = actions["uav_0"]
            
        if uav_action is not None:
            speed = np.linalg.norm(uav_action)
            if speed > EnvConfig.UAV_SPEED:
                uav_action = (uav_action / speed) * EnvConfig.UAV_SPEED
            
            self.uav.vx = uav_action[0]
            self.uav.vy = uav_action[1]
            self.uav.x += self.uav.vx * self.dt
            self.uav.y += self.uav.vy * self.dt
            
            self.uav.x = np.clip(self.uav.x, 0, self.area_size)
            self.uav.y = np.clip(self.uav.y, 0, self.area_size)
            
            v_uav = self.uav.velocity_magnitude
            self.uav.consume_energy(v_uav, self.dt)

        # Jammer
        jam_power_out = 0.0
        jam_total_power_cost = 0.0
        
        if "jammer_0" in actions:
            jam_action = actions["jammer_0"]
            if self.flatten_actions:
                # Discrete(30) -> [Channel, PowerLevel]
                # ch = action // 10 (0,1,2), p_level = action % 10 (0-9)
                ch_idx = int(jam_action // 10)
                p_level = int(jam_action % 10)
            else:
                # MultiDiscrete: [Channel, PowerLevel]
                ch_idx = jam_action[0]
                p_level = jam_action[1]
            
            # Update Entity State
            self.attacker.current_channel = ch_idx
            
            # Map Power: Level 0..9 -> 0..Max
            jam_power_out = (p_level / 9.0) * EnvConfig.MAX_JAMMING_POWER
            self.attacker.set_jamming_power(jam_power_out)
            
            # Calculate Consumption Cost (Cui et al.)
            freq = UAVConfig.CHANNELS[ch_idx]
            jam_total_power_cost = physics.calculate_jammer_power_consumption(jam_power_out, freq)
            
        # Nodes (No action)

        # 2. Physics & Logic Interactions
        step_log = {
            "step": self.current_step,
            "uav_x": self.uav.x,
            "uav_y": self.uav.y,
            "jammer_power": self.attacker.jamming_power,
            "uav_energy": self.uav.total_energy_consumed,
            "uav_channel": self.uav.current_channel,
            "jammer_channel": self.attacker.current_channel
        }

        total_sinr = 0
        jammed_count = 0
        
        for i, node in enumerate(self.nodes):
            # Update Node Channel (Assume synced with UAV for reception)
            node.current_channel = self.uav.current_channel
            
            dist_uav = np.linalg.norm(node.position - self.uav.position)
            dist_jam = np.linalg.norm(node.position - self.attacker.position)
            
            # Frequency dependent Path Loss
            freq = UAVConfig.CHANNELS[self.uav.current_channel]
            beta_uav = physics.calculate_path_loss(dist_uav, fc=freq)
            beta_jam = physics.calculate_path_loss(dist_jam, fc=freq) # Jammer distance on same freq
            
            p_rx_signal = physics.calculate_received_power(node.tx_power, dist_uav, beta_uav)
            
            # Interference Check: Only if Channels Match
            p_rx_jam = 0.0
            if self.uav.current_channel == self.attacker.current_channel:
                 p_rx_jam = physics.calculate_received_power(self.attacker.jamming_power, dist_jam, beta_jam)
            
            sinr = physics.calculate_sinr(p_rx_signal, UAVConfig.N0_Linear, p_rx_jam)
            
            # Theoretical (No Jam)
            sinr_no_jam = physics.calculate_sinr(p_rx_signal, UAVConfig.N0_Linear, 0.0)
            
            rate = physics.calculate_data_rate(UAVConfig.B, sinr)
            
            connected_now = sinr > UAVConfig.SINR_THRESHOLD
            connected_theoretical = sinr_no_jam > UAVConfig.SINR_THRESHOLD
            
            status = 0
            if connected_now:
                status = 0
            else:
                if not connected_theoretical:
                    status = 1
                else:
                    status = 2
                    jammed_count += 1
            
            node.connection_status = status
            node.update_aoi(self.dt, success=(status == 0))
            node.consume_energy(rate)
            
            step_log[f"node_{i}_status"] = status
            step_log[f"node_{i}_sinr"] = sinr
            step_log[f"node_{i}_aoi"] = node.aoi
            # Log coordinates for Visualization (Static but needed by Visualizer parsing)
            step_log[f"node_{i}_x"] = node.x
            step_log[f"node_{i}_y"] = node.y

        # 3. Rewards
        # Jammer Reward: Success - Cost
        # Scaling: +10 per jammed node, -0.1 per Watt cost (Example)
        reward_jam_success = float(jammed_count) * 10
        reward_energy_cost = jam_total_power_cost * 0.1
        
        # 3. Rewards
        # Jammer Reward: Success - Cost
        
        # A. Success Reward (High sparse reward)
        reward_jam_success = float(jammed_count) * 10
        
        # B. Cost Penalty (Low dense penalty)
        reward_energy_cost = jam_total_power_cost * 0.1
        
        # C. Tracking Reward (Dense guidance)
        # Only reward tracking if jammer is ACTUALLY using power
        # This prevents zero-power exploitation strategy
        reward_tracking = 0.0
        if (self.attacker.current_channel == self.uav.current_channel 
            and self.attacker.jamming_power > 0.01):  # Minimum power threshold
            reward_tracking = 0.5 
        
        jammer_reward = reward_jam_success + reward_tracking - reward_energy_cost
        uav_reward = -float(jammed_count) 
        
        rewards = {}
        terminated = self.current_step >= self.max_steps
        truncations = {agent: False for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        
        for agent in self.agents:
            if agent == "jammer_0":
                rewards[agent] = jammer_reward
            elif agent == "uav_0":
                rewards[agent] = uav_reward
            else:
                rewards[agent] = 0.0

        # 4. Logging & Obs
        step_log["jammed_count"] = jammed_count
        step_log["jammer_cost"] = jam_total_power_cost
        if self.logger:
            self.logger.log_step(step_log)

        observations = {agent: self._get_obs() for agent in self.agents}
        infos = {agent: step_log for agent in self.agents}

        if terminated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self):
        # Construct global observation vector
        # Normalization Factor
        norm_scale = self.area_size
        
        obs = []
        
        # SENSING Based Observation (RSSI Proxy)
        # Instead of Absolute Coordinates, we give Distance.
        dist = np.linalg.norm(self.uav.position - self.attacker.position)
        norm_dist = dist / norm_scale
        obs.append(norm_dist)
        
        # Channels 
        obs.extend([float(self.uav.current_channel), float(self.attacker.current_channel)])
        
        sinrs = []
        aois = []
        for node in self.nodes:
            # Jammer knows static node locations (Map)
            obs.extend([node.x / norm_scale, node.y / norm_scale])
            aois.append(node.aoi) 
            # sinrs.append(0.0) # Removed SINR from INPUT to reduce noise/dim
            
        # obs.extend(sinrs)
        obs.extend(aois)
        
        return np.array(obs, dtype=np.float32)

    def render(self):
        pass

    def state(self):
        return self._get_obs()
