import numpy as np
from confs.config import UAVConfig
from confs.env_config import EnvConfig
# Type hinting only, avoid circular import at runtime if possible or use simple typing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation.pettingzoo_env import UAV_IoT_PZ_Env

class UAVRuleBasedController:
    """
    Control logic for the UAV agent.
    Implements cooperative dynamic waypoint navigation for multiple UAVs.
    """
    def __init__(self, env, uav_id: int = 0):
        self.env = env # Store reference to environment
        self.uav_id = uav_id
        self.target_node = None # Current target node object
        self.area_size = EnvConfig.AREA_SIZE
        self.dt = EnvConfig.STEP_TIME
        
        # Hover state
        self.hover_timer = 0.0
        self.hover_duration = 5.0 # Seconds
        self.is_hovering = False
        
        # Markov Channel Hopping State
        from confs.config import UAVConfig
        self.num_channels = len(UAVConfig.CHANNELS)
        
        # Determine transition matrix dynamically or use preset for C=3
        if self.num_channels == 3:
            self.transition_matrix = np.array([
                [0.1, 0.7, 0.2], # From Ch0 -> Prefer Ch1
                [0.2, 0.1, 0.7], # From Ch1 -> Prefer Ch2
                [0.7, 0.2, 0.1]  # From Ch2 -> Prefer Ch0
            ])
        else:
            # Dynamic C x C Markov transition matrix:
            self.transition_matrix = np.zeros((self.num_channels, self.num_channels))
            for i in range(self.num_channels):
                self.transition_matrix[i, i] = 0.1
                next_ch = (i + 1) % self.num_channels
                self.transition_matrix[i, next_ch] = 0.6
                remaining_prob = 0.3
                remaining_channels = [ch for ch in range(self.num_channels) if ch != i and ch != next_ch]
                if remaining_channels:
                    prob_per_channel = remaining_prob / len(remaining_channels)
                    for r_ch in remaining_channels:
                        self.transition_matrix[i, r_ch] = prob_per_channel
                else:
                    self.transition_matrix[i, next_ch] += remaining_prob

        # SINR/Jamming Response Logic
        self.last_step_jammed = False
        self.jammed_step_counter = 0
        self.persistence_threshold = UAVConfig.PERSISTENCE_THRESHOLD # Steps to wait before channel-switching

    def update_channel_logic(self):
        """
        Markov Decision Process for Channel Selection.
        Called every step.
        """
        uav = self.env.uavs[self.uav_id]
        target_node = self.target_node
        
        if target_node is None:
            return
        
        # Check if current target node is Jammed (Status 2)
        is_jammed = (target_node.connection_status == 2)
        
        if is_jammed:
            self.jammed_step_counter += 1
        else:
            self.jammed_step_counter = 0
            
        # Only switch if jammed for N consecutive steps
        if is_jammed and self.jammed_step_counter >= self.persistence_threshold:
            # Trigger Channel Switch
            current_ch = uav.current_channel
            probs = self.transition_matrix[current_ch]
            next_ch = np.random.choice(np.arange(self.num_channels), p=probs)
            uav.current_channel = next_ch
            self.jammed_step_counter = 0
            
        self.last_step_jammed = is_jammed

    def get_action(self, observation=None) -> np.ndarray:
        """
        Determines the velocity vector (vx, vy) for the UAV.
        Implements moving to waypoint -> hovering -> moving to next.
        """
        uav = self.env.uavs[self.uav_id]
        nodes = self.env.nodes
        
        if uav is None or not nodes:
            return np.zeros(2, dtype=np.float32)

        # Logic
        self.update_channel_logic()

        if self.is_hovering:
            self.hover_timer += self.dt
            if self.hover_timer >= self.hover_duration:
                # Finished hovering, remove node from unvisited pool
                if self.target_node is not None:
                    self.env.unvisited_nodes.discard(self.target_node.id)
                self.is_hovering = False
                self.hover_timer = 0.0
                self.target_node = None
                if self.uav_id in self.env.targeted_nodes:
                    del self.env.targeted_nodes[self.uav_id]
                
                # Select next target
                self._select_next_target()
            else:
                # Stay hovering
                return np.zeros(2, dtype=np.float32)

        # Navigation Mode
        if self.target_node is None:
            self._select_next_target()
            if self.target_node is None:
                return np.zeros(2, dtype=np.float32)

        target_pos = np.array([self.target_node.x, self.target_node.y, uav.z])
        current_pos = np.array([uav.x, uav.y, uav.z])
        
        direction_vector = target_pos - current_pos
        dist_to_target = np.linalg.norm(direction_vector[:2]) # XY distance
        
        # Dynamic threshold
        step_distance = EnvConfig.UAV_SPEED * self.dt
        arrival_threshold = max(10.0, step_distance * 1.1) 
        
        # Check if reached
        if dist_to_target < arrival_threshold:
            # Reached target, start hovering
            self.is_hovering = True
            self.hover_timer = 0.0
            return np.zeros(2, dtype=np.float32)
            
        speed = EnvConfig.UAV_SPEED
        
        if dist_to_target > 0:
            norm_dir = direction_vector[:2] / np.linalg.norm(direction_vector[:2])
            velocity_vector = norm_dir * speed
        else:
            velocity_vector = np.zeros(2)
            
        # APF Repulsion disabled to prevent computational delay and keep trajectory direct
        final_velocity = velocity_vector
        return final_velocity.astype(np.float32)

    def _select_next_target(self):
        """
        Cooperative Waypoint Selection.
        """
        unvisited = [node for node in self.env.nodes if node.id in self.env.unvisited_nodes]
        targeted_by_others = {t.id for uid, t in self.env.targeted_nodes.items() if uid != self.uav_id and t is not None}
        available = [node for node in unvisited if node.id not in targeted_by_others]
        
        if not available:
            if self.env.unvisited_nodes:
                available = unvisited
            else:
                # All nodes visited. Reset unvisited pool.
                if self.uav_id == 0:
                    self.env.unvisited_nodes = set(range(len(self.env.nodes)))
                unvisited = self.env.nodes
                available = [node for node in unvisited if node.id not in targeted_by_others]
                if not available:
                    available = unvisited
                    
        if available:
            uav = self.env.uavs[self.uav_id]
            current_pos = np.array([uav.x, uav.y])
            closest_node = min(available, key=lambda n: np.linalg.norm(np.array([n.x, n.y]) - current_pos))
            self.target_node = closest_node
            self.env.targeted_nodes[self.uav_id] = closest_node
        else:
            self.target_node = None

    def reset(self):
        self.hover_timer = 0.0
        self.is_hovering = False
        self.target_node = None
        self._select_next_target()
