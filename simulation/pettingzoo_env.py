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
        # If auto_uav is True, UAVs are part of environment (not agents)
        if self.auto_uav:
            self.possible_agents = ["jammer_0"] + [f"node_{i}" for i in range(EnvConfig.NUM_NODES)]
        else:
            self.possible_agents = [f"uav_{i}" for i in range(EnvConfig.NUM_UAVS)] + ["jammer_0"] + [f"node_{i}" for i in range(EnvConfig.NUM_NODES)]
            
        self.agents = self.possible_agents[:]
        
        # --- Entities ---
        self.uavs = []
        self.attacker = None
        self.nodes = []
        self.uav_controllers = []

        # --- Spaces ---
        self.observation_spaces = {}
        self.action_spaces = {}
        
        # Common Observation Space Dimension queried dynamically from EnvConfig
        obs_dim = EnvConfig.get_obs_dim()
        common_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        for agent in self.possible_agents:
            self.observation_spaces[agent] = common_obs_space
            
            if agent.startswith("uav_"):
                max_speed = EnvConfig.UAV_SPEED * 2 
                self.action_spaces[agent] = spaces.Box(low=-max_speed, high=max_speed, shape=(2,), dtype=np.float32)
            
            elif agent == "jammer_0":
                num_ch = len(UAVConfig.CHANNELS)
                if self.flatten_actions:
                    self.action_spaces[agent] = spaces.Discrete(num_ch * 10)
                else:
                    self.action_spaces[agent] = spaces.MultiDiscrete([num_ch, 10])
            
            else: # Nodes
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
        
        # Reset Entities with spaced-out positions
        self.uavs = []
        for i in range(EnvConfig.NUM_UAVS):
            spacing = (i + 1) * (self.area_size / (EnvConfig.NUM_UAVS + 1))
            uav = UAVAgent(x=spacing, y=spacing, z=EnvConfig.UAV_START_Z)
            self.uavs.append(uav)
            
        self.attacker = SmartAttacker(x=EnvConfig.ATTACKER_POS_X, y=EnvConfig.ATTACKER_POS_Y)
        
        self.nodes = []
        if seed is not None:
            np.random.seed(seed)
            
        for i in range(EnvConfig.NUM_NODES):
            nx = np.random.uniform(0, self.area_size)
            ny = np.random.uniform(0, self.area_size)
            self.nodes.append(IoTNode(i, nx, ny))

        # Dynamic coordination states
        self.unvisited_nodes = set(range(EnvConfig.NUM_NODES))
        self.targeted_nodes = {}

        if self.auto_uav:
            self.uav_controllers = [UAVRuleBasedController(self, uav_id=i) for i in range(EnvConfig.NUM_UAVS)]
            for ctrl in self.uav_controllers:
                ctrl.reset()

        self._cached_obs = self._get_obs()
        observations = {agent: self._cached_obs for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        self.current_step += 1
        
        # 1. Apply Actions
        # UAVs
        for i in range(EnvConfig.NUM_UAVS):
            uav = self.uavs[i]
            uav_action = None
            
            if self.auto_uav:
                uav_action = self.uav_controllers[i].get_action()
            elif f"uav_{i}" in actions:
                uav_action = actions[f"uav_{i}"]
                
            if uav_action is not None:
                speed = np.linalg.norm(uav_action)
                if speed > EnvConfig.UAV_SPEED:
                    uav_action = (uav_action / speed) * EnvConfig.UAV_SPEED
                
                uav.vx = uav_action[0]
                uav.vy = uav_action[1]
                uav.x += uav.vx * self.dt
                uav.y += uav.vy * self.dt
                
                uav.x = np.clip(uav.x, 0, self.area_size)
                uav.y = np.clip(uav.y, 0, self.area_size)
                
                v_uav = uav.velocity_magnitude
                uav.consume_energy(v_uav, self.dt)

        # Jammer
        jam_power_out = 0.0
        jam_total_power_cost = 0.0
        
        if "jammer_0" in actions:
            jam_action = actions["jammer_0"]
            ch_idx, p_level = EnvConfig.parse_jammer_action(jam_action, self.flatten_actions)
            
            self.attacker.current_channel = ch_idx
            jam_power_out = (p_level / 9.0) * EnvConfig.MAX_JAMMING_POWER
            self.attacker.set_jamming_power(jam_power_out)
            
            freq = UAVConfig.CHANNELS[ch_idx]
            jam_total_power_cost = physics.calculate_jammer_power_consumption(jam_power_out, freq)

        # 2. Physics & Logic Interactions
        step_log = {
            "step": self.current_step,
            "jammer_power": self.attacker.jamming_power,
            "jammer_channel": self.attacker.current_channel,
            "jammer_cost": jam_total_power_cost
        }
        
        # Find closest UAV to jammer for tracking stats/reward
        closest_uav = min(self.uavs, key=lambda uav: np.linalg.norm(uav.position - self.attacker.position))
        
        # Keep backward compatibility with single-uav plotting
        step_log["uav_x"] = self.uavs[0].x
        step_log["uav_y"] = self.uavs[0].y
        step_log["uav_energy"] = self.uavs[0].total_energy_consumed
        step_log["uav_channel"] = closest_uav.current_channel
        
        # Log all UAVs individually
        for m, uav in enumerate(self.uavs):
            step_log[f"uav_{m}_x"] = uav.x
            step_log[f"uav_{m}_y"] = uav.y
            step_log[f"uav_{m}_energy"] = uav.total_energy_consumed
            step_log[f"uav_{m}_channel"] = uav.current_channel

        total_sinr = 0
        jammed_count = 0
        
        # Precompute pairwise distance matrix between all nodes and all UAVs to optimize step speed
        node_uav_dists = []
        for node in self.nodes:
            dists = [np.linalg.norm(node.position - uav.position) for uav in self.uavs]
            node_uav_dists.append(dists)
            
        node_assoc_info = []
        for idx, dists in enumerate(node_uav_dists):
            assoc_uav_idx = int(np.argmin(dists))
            node_assoc_info.append((assoc_uav_idx, dists[assoc_uav_idx]))
            
        for i, node in enumerate(self.nodes):
            # Dynamic Node Association (closest UAV) - read from precomputed info
            assoc_uav_idx, dist_uav = node_assoc_info[i]
            assoc_uav = self.uavs[assoc_uav_idx]
            dist_jam = np.linalg.norm(node.position - self.attacker.position)
            
            node.current_channel = assoc_uav.current_channel
            freq = UAVConfig.CHANNELS[assoc_uav.current_channel]
            beta_uav = physics.calculate_path_loss(dist_uav, fc=freq)
            beta_jam = physics.calculate_path_loss(dist_jam, fc=freq)
            
            # Check if this node is transmitting (only when a UAV is hovering over it to gather data)
            is_transmitting = False
            trans_uav_idx = -1
            if self.auto_uav and hasattr(self, 'uav_controllers'):
                for u_idx, ctrl in enumerate(self.uav_controllers):
                    if ctrl.is_hovering and ctrl.target_node and ctrl.target_node.id == node.id:
                        is_transmitting = True
                        trans_uav_idx = u_idx
                        break
            else:
                # Fallback: assume transmitting if a UAV is within arrival threshold (approx 15m)
                for u_idx, uav in enumerate(self.uavs):
                    if np.linalg.norm(node.position - uav.position) < 15.0:
                        is_transmitting = True
                        trans_uav_idx = u_idx
                        break
            
            if is_transmitting:
                p_rx_signal = physics.calculate_received_power(node.tx_power, dist_uav, beta_uav)
                
                p_rx_jam = 0.0
                if assoc_uav.current_channel == self.attacker.current_channel:
                     p_rx_jam = physics.calculate_received_power(self.attacker.jamming_power, dist_jam, beta_jam)
                
                # Co-channel interference only from other transmitting nodes on the same channel
                p_rx_co_channel = 0.0
                for k, other_node in enumerate(self.nodes):
                    if k == i:
                        continue
                    
                    other_transmitting = False
                    other_assoc_idx = -1
                    if self.auto_uav and hasattr(self, 'uav_controllers'):
                        for u_idx, ctrl in enumerate(self.uav_controllers):
                            if ctrl.is_hovering and ctrl.target_node and ctrl.target_node.id == other_node.id:
                                other_transmitting = True
                                other_assoc_idx = u_idx
                                break
                    else:
                        for u_idx, uav in enumerate(self.uavs):
                            if np.linalg.norm(other_node.position - uav.position) < 15.0:
                                other_transmitting = True
                                other_assoc_idx = u_idx
                                break
                    
                    if other_transmitting:
                        other_assoc_uav = self.uavs[other_assoc_idx]
                        if other_assoc_uav.current_channel == assoc_uav.current_channel:
                            dist_other_to_uav = node_uav_dists[k][assoc_uav_idx]
                            beta_other = physics.calculate_path_loss(dist_other_to_uav, fc=freq)
                            p_rx_co_channel += physics.calculate_received_power(other_node.tx_power, dist_other_to_uav, beta_other)
                
                sinr = physics.calculate_sinr(p_rx_signal, UAVConfig.N0_Linear, p_rx_jam + p_rx_co_channel)
                sinr_no_jam = physics.calculate_sinr(p_rx_signal, UAVConfig.N0_Linear, p_rx_co_channel)
                
                rate = physics.calculate_data_rate(UAVConfig.B, sinr)
                
                sinr_threshold_linear = 10 ** (UAVConfig.SINR_THRESHOLD / 10.0)
                connected_now = sinr > sinr_threshold_linear
                connected_theoretical = sinr_no_jam > sinr_threshold_linear
                
                status = 0
                if connected_now:
                    status = 0
                else:
                    if not connected_theoretical:
                        status = 1
                    else:
                        status = 2
                        jammed_count += 1
            else:
                status = 1  # Out of range if UAV is not collecting data
                sinr = 0.0
                rate = 0.0
            
            node.connection_status = status
            node.update_aoi(self.dt, success=(status == 0))
            node.consume_energy(rate)
            
            step_log[f"node_{i}_status"] = status
            step_log[f"node_{i}_sinr"] = sinr
            step_log[f"node_{i}_aoi"] = node.aoi
            step_log[f"node_{i}_x"] = node.x
            step_log[f"node_{i}_y"] = node.y

        # 3. Rewards
        # Jammer Reward
        reward_jam_success = float(jammed_count) * EnvConfig.W_SUCCESS
        reward_energy_cost = jam_total_power_cost * EnvConfig.W_COST
        
        # Dense Tracking Reward: Jammer matches the channel of the closest UAV
        reward_tracking = 0.0
        closest_uav = min(self.uavs, key=lambda uav: np.linalg.norm(uav.position - self.attacker.position))
        if self.attacker.current_channel == closest_uav.current_channel and self.attacker.jamming_power > 0.01:
            reward_tracking = EnvConfig.W_TRACKING
        
        jammer_reward = reward_jam_success + reward_tracking - reward_energy_cost
        uav_reward = -float(jammed_count)
        
        rewards = {}
        terminated = self.current_step >= self.max_steps
        truncations = {agent: False for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        
        for agent in self.agents:
            if agent == "jammer_0":
                rewards[agent] = jammer_reward
            elif agent.startswith("uav_"):
                rewards[agent] = uav_reward
            else:
                rewards[agent] = 0.0

        step_log["jammed_count"] = jammed_count
        if self.logger:
            self.logger.log_step(step_log)

        # Clear old cache and compute new observation once for all agents
        self._cached_obs = None
        self._cached_obs = self._get_obs()
        observations = {agent: self._cached_obs for agent in self.agents}
        infos = {agent: step_log for agent in self.agents}

        if terminated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self):
        if hasattr(self, '_cached_obs') and self._cached_obs is not None:
            return self._cached_obs
        self._cached_obs = self._get_obs_vector()
        return self._cached_obs

    def _get_obs_vector(self):
        norm_scale = self.area_size
        
        # Distances from Jammer to all UAVs
        uav_distances = []
        for uav in self.uavs:
            dist = np.linalg.norm(uav.position - self.attacker.position)
            uav_distances.append(dist / norm_scale)
        
        # Channels: each UAV's channel
        uav_channels = [uav.current_channel for uav in self.uavs]
        
        # Jammer channel
        jammer_channel = self.attacker.current_channel
        
        # Nodes: RSSI at Jammer (replacing coordinates)
        node_rssis = []
        for node in self.nodes:
            # Calculate distance to jammer
            dist_jam = np.linalg.norm(node.position - self.attacker.position)
            
            # Check if this node is transmitting (a UAV is hovering over it to gather data)
            is_transmitting = False
            if self.auto_uav and hasattr(self, 'uav_controllers'):
                for ctrl in self.uav_controllers:
                    if ctrl.is_hovering and ctrl.target_node and ctrl.target_node.id == node.id:
                        is_transmitting = True
                        break
            else:
                for uav in self.uavs:
                    if np.linalg.norm(node.position - uav.position) < 15.0:
                        is_transmitting = True
                        break
            
            if is_transmitting:
                # Find closest UAV to get its channel frequency
                dists = [np.linalg.norm(node.position - uav.position) for uav in self.uavs]
                assoc_uav_idx = int(np.argmin(dists))
                assoc_uav = self.uavs[assoc_uav_idx]
                freq = UAVConfig.CHANNELS[assoc_uav.current_channel]
                
                beta_jam = physics.calculate_path_loss(dist_jam, fc=freq)
                p_rx = physics.calculate_received_power(node.tx_power, dist_jam, beta_jam)
                # Avoid exact zero or negative in log
                p_rx = max(p_rx, 1e-30)
                rssi = 10 * np.log10(p_rx / 1e-3)
            else:
                rssi = -150.0 # Noise floor / inactive
            
            # Normalize RSSI to [0, 1] range: -150 dBm -> 0.0, -50 dBm or higher -> 1.0
            rssi_norm = np.clip((rssi + 150.0) / 100.0, 0.0, 1.0)
            node_rssis.append(rssi_norm)
        
        return EnvConfig.build_state_vector(
            uav_distances=uav_distances,
            uav_channels=uav_channels,
            jammer_channel=jammer_channel,
            node_rssis=node_rssis
        )

    def render(self):
        pass

    def state(self):
        return self._get_obs()
