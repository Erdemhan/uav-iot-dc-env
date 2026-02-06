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

    def get_action(self, observation=None) -> np.ndarray:
        """
        Determines the velocity vector (vx, vy) for the UAV to reach the next waypoint.
        Returns:
            np.ndarray: [vx, vy]
        """
        uav = self.env.uav
        nodes = self.env.nodes
        
        if uav is None:
            return np.zeros(2, dtype=np.float32)

        if not nodes:
            # Fallback if no nodes
            target_pos = np.array([self.area_size/2, self.area_size/2, UAVConfig.H])
        else:
            target_node = nodes[self.target_node_index]
            target_pos = np.array([target_node.x, target_node.y, uav.z])
            
        current_pos = np.array([uav.x, uav.y, uav.z])
        direction_vector = target_pos - current_pos
        dist_to_target = np.linalg.norm(direction_vector[:2]) # XY distance
        
        # Dynamic threshold to prevent overshooting
        # If step is 25m (5m/s * 5s), threshold must be > 12.5m or ideally > 25m to guarantee capture.
        step_distance = EnvConfig.UAV_SPEED * self.dt
        arrival_threshold = max(10.0, step_distance * 1.1) 
        
        # Check if reached
        if dist_to_target < arrival_threshold:
            # Switch to next node
            self.target_node_index = (self.target_node_index + 1) % len(nodes)
            # Recalculate for new target
            target_node = nodes[self.target_node_index]
            target_pos = np.array([target_node.x, target_node.y, uav.z])
            direction_vector = target_pos - current_pos
            dist_to_target = np.linalg.norm(direction_vector[:2])
            
        speed = EnvConfig.UAV_SPEED
        
        if dist_to_target > 0:
            norm_dir = direction_vector / np.linalg.norm(direction_vector)
            velocity_vector = norm_dir * speed
        else:
            velocity_vector = np.zeros(3)
            
        # We return only vx, vy as action
        return velocity_vector[:2].astype(np.float32)

    def reset(self):
        self.target_node_index = 0
