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
    Implements the waypoint navigation to visit nodes in a sequence.
    """
    def __init__(self, env):
        self.env = env # Store reference to environment
        self.target_node_index = 0
        self.area_size = EnvConfig.AREA_SIZE
        self.dt = EnvConfig.STEP_TIME
        
        # Hover state
        self.hover_timer = 0.0
        self.hover_duration = 5.0 # Seconds
        self.is_hovering = False

    def get_action(self, observation=None) -> np.ndarray:
        """
        Determines the velocity vector (vx, vy) for the UAV.
        Implements moving to waypoint -> hovering -> moving to next.
        """
        uav = self.env.uav
        nodes = self.env.nodes
        
        if uav is None:
            return np.zeros(2, dtype=np.float32)

        if not nodes:
            # Fallback if no nodes
            return np.zeros(2, dtype=np.float32)

        # Logic
        if self.is_hovering:
            self.hover_timer += self.dt
            if self.hover_timer >= self.hover_duration:
                # Finished hovering, move to next node
                self.is_hovering = False
                self.hover_timer = 0.0
                self.target_node_index = (self.target_node_index + 1) % len(nodes)
            else:
                # Stay hovering
                return np.zeros(2, dtype=np.float32)

        # Navigation Mode
        target_node = nodes[self.target_node_index]
        target_pos = np.array([target_node.x, target_node.y, uav.z])
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
            # Stop immediately
            return np.zeros(2, dtype=np.float32)
            
        speed = EnvConfig.UAV_SPEED
        
        if dist_to_target > 0:
            norm_dir = direction_vector / np.linalg.norm(direction_vector)
            velocity_vector = norm_dir * speed
        else:
            velocity_vector = np.zeros(3)
            
        return velocity_vector[:2].astype(np.float32)

    def reset(self):
        self.target_node_index = 0
        self.hover_timer = 0.0
        self.is_hovering = False
