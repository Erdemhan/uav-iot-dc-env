from pettingzoo.utils import ParallelEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from confs.config import UAVConfig
from confs.env_config import EnvConfig
from simulation.entities import UAVAgent, IoTNode, SmartAttacker
import core.physics as physics
from core.logger import SimulationLogger

class UAV_IoT_PZ_Env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "uav_iot_v1"}

    def __init__(self, logger: SimulationLogger = None):
        super().__init__()
        self.logger = logger
        self.area_size = EnvConfig.AREA_SIZE
        
        # --- Agents Configuration ---
        self.possible_agents = ["uav_0", "jammer_0"] + [f"node_{i}" for i in range(EnvConfig.NUM_NODES)]
        self.agents = self.possible_agents[:]
        
        # --- Entities ---
        # These will be reset in self.reset()
        self.uav = None
        self.attacker = None
        self.nodes = []

        # --- Spaces ---
        self.observation_spaces = {}
        self.action_spaces = {}
        
        # Common Observation Space Dimension
        # UAV (x, y), Attacker (x, y), Nodes (x, y) * N, Node SINR * N, Node AoI * N
        obs_dim = 4 + 4 * EnvConfig.NUM_NODES
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
                # Action: Jamming Power [P_jam]
                self.action_spaces[agent] = spaces.Box(low=0.0, high=EnvConfig.MAX_JAMMING_POWER, shape=(1,), dtype=np.float32)
            
            else: # Nodes
                # Action: No-Op (Discrete(1))
                self.action_spaces[agent] = spaces.Discrete(1)

        self.current_step = 0
        self.max_steps = EnvConfig.MAX_STEPS
        self.dt = EnvConfig.STEP_TIME

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

        observations = {agent: self._get_obs() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        self.current_step += 1
        
        # 1. Apply Actions
        # UAV
        if "uav_0" in actions:
            uav_action = actions["uav_0"]
            # Apply velocity update
            # uav_action is [vx, vy]. 
            # We calculate new position based on this velocity.
            # In the rule-based controller transparency, we trust the controller gives valid velocity.
            # But we should clamp to max speed just in case.
            speed = np.linalg.norm(uav_action)
            if speed > EnvConfig.UAV_SPEED:
                uav_action = (uav_action / speed) * EnvConfig.UAV_SPEED
            
            # Simple Euler Integration
            self.uav.vx = uav_action[0]
            self.uav.vy = uav_action[1]
            self.uav.x += self.uav.vx * self.dt
            self.uav.y += self.uav.vy * self.dt
            
            # Boundary checks (Optional but good practice)
            self.uav.x = np.clip(self.uav.x, 0, self.area_size)
            self.uav.y = np.clip(self.uav.y, 0, self.area_size)
            
            # Energy
            v_uav = self.uav.velocity_magnitude
            self.uav.consume_energy(v_uav, self.dt)

        # Jammer
        if "jammer_0" in actions:
            jam_action = actions["jammer_0"]
            self.attacker.set_jamming_power(float(jam_action[0]))
            
        # Nodes (No action needed as they are static/passive mostly)

        # 2. Physics & Logic Interactions
        step_log = {
            "step": self.current_step,
            "uav_x": self.uav.x,
            "uav_y": self.uav.y,
            "jammer_power": self.attacker.jamming_power,
            "uav_energy": self.uav.total_energy_consumed
        }

        total_sinr = 0
        jammed_count = 0
        
        for i, node in enumerate(self.nodes):
            dist_uav = np.linalg.norm(node.position - self.uav.position)
            dist_jam = np.linalg.norm(node.position - self.attacker.position)
            
            beta_uav = physics.calculate_path_loss(dist_uav)
            beta_jam = physics.calculate_path_loss(dist_jam)
            
            p_rx_signal = physics.calculate_received_power(node.tx_power, dist_uav, beta_uav)
            p_rx_jam = physics.calculate_received_power(self.attacker.jamming_power, dist_jam, beta_jam)
            
            sinr = physics.calculate_sinr(p_rx_signal, UAVConfig.N0_Linear, p_rx_jam)
            sinr_no_jam = physics.calculate_sinr(p_rx_signal, UAVConfig.N0_Linear, 0.0)
            
            rate = physics.calculate_data_rate(UAVConfig.B, sinr)
            
            connected_now = sinr > 1.0
            connected_theoretical = sinr_no_jam > 1.0
            
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
            
            step_log[f"node_{i}_x"] = node.x
            step_log[f"node_{i}_y"] = node.y
            step_log[f"node_{i}_sinr"] = sinr
            step_log[f"node_{i}_aoi"] = node.aoi
            step_log[f"node_{i}_energy"] = node.total_energy_consumed
            step_log[f"node_{i}_status"] = status
            step_log[f"node_{i}_total_time"] = node.total_connected_duration
            step_log[f"node_{i}_max_continuous_time"] = node.max_continuous_duration

        # 3. Rewards & Termination
        # Reward: Global reward for Jammer is maximize jammed nodes.
        # For this environment, we can define rewards per agent.
        # Jammer: +1 per jammed node.
        # UAV: -1 per jammed node (Cooperative with nodes), or Minimize AoI.
        # Nodes: Minimize AoI.
        
        # Important: The user asked to keep Jammer as Learning agent, others Rule-based.
        # We assign rewards regardless, mostly for Jammer.
        
        rewards = {}
        terminated = self.current_step >= self.max_steps
        truncations = {agent: False for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        
        jammer_reward = float(jammed_count)
        # Simple placeholder rewards for others
        uav_reward = -float(jammed_count) 
        node_reward = 0.0 
        
        for agent in self.agents:
            if agent == "jammer_0":
                rewards[agent] = jammer_reward
            elif agent == "uav_0":
                rewards[agent] = uav_reward
            else:
                rewards[agent] = node_reward

        # 4. Logging
        step_log["jammed_count"] = jammed_count
        if self.logger:
            self.logger.log_step(step_log)

        # 5. Get Observations
        observations = {agent: self._get_obs() for agent in self.agents}
        infos = {agent: step_log for agent in self.agents} # Sharing logs in info

        if terminated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self):
        # Construct global observation vector
        obs = []
        obs.extend([self.uav.x, self.uav.y])
        obs.extend([self.attacker.x, self.attacker.y])
        
        sinrs = []
        aois = []
        for node in self.nodes:
            obs.extend([node.x, node.y])
            aois.append(node.aoi)
            # Re-calculating SINR here might be redundant but strictly accurate for "observation"
            # We can approximate with last step logic or just re-calc.
            # For efficiency let's re-calc briefly or store in node.
            # Using stored variables from step would be better design but let's re-calc for stateless safety
            dist_uav = np.linalg.norm(node.position - self.uav.position)
            dist_jam = np.linalg.norm(node.position - self.attacker.position)
            beta_uav = physics.calculate_path_loss(dist_uav)
            beta_jam = physics.calculate_path_loss(dist_jam)
            p_rx = physics.calculate_received_power(node.tx_power, dist_uav, beta_uav)
            p_jam = physics.calculate_received_power(self.attacker.jamming_power, dist_jam, beta_jam)
            val_sinr = physics.calculate_sinr(p_rx, UAVConfig.N0_Linear, p_jam)
            sinrs.append(val_sinr)
            
        obs.extend(sinrs)
        obs.extend(aois)
        
        return np.array(obs, dtype=np.float32)

    def render(self):
        pass

    def state(self):
        return self._get_obs()
