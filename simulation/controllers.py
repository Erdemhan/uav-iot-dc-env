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
        self.target_node = None # Current target node object
        self.waypoint_queue = [] # List of nodes to visit in order
        self.area_size = EnvConfig.AREA_SIZE
        self.dt = EnvConfig.STEP_TIME
        
        # Hover state
        self.hover_timer = 0.0
        self.hover_duration = 5.0 # Seconds
        self.is_hovering = False
        
        # Markov Channel Hopping State
        self.num_channels = 3
        self.transition_matrix = np.array([
            [0.1, 0.7, 0.2], # From Ch0 -> Prefer Ch1
            [0.2, 0.1, 0.7], # From Ch1 -> Prefer Ch2
            [0.7, 0.2, 0.1]  # From Ch2 -> Prefer Ch0
        ])
        # We need to track if we already switched for this jamming instance to avoid rapid switching
        self.last_step_jammed = False

    def update_channel_logic(self):
        """
        Markov Decision Process for Channel Selection.
        Called every step.
        """
        uav = self.env.uav
        target_node = self.target_node
        
        if target_node is None:
            return
        
        # Check if current target node is Jammed (Status 2)
        # We access the entity state directly.
        is_jammed = (target_node.connection_status == 2)
        
        if is_jammed and not self.last_step_jammed:
            # Trigger Channel Switch
            current_ch = uav.current_channel
            probs = self.transition_matrix[current_ch]
            next_ch = np.random.choice(np.arange(self.num_channels), p=probs)
            
            uav.current_channel = next_ch
            # print(f"DEBUG: UAV Hopped {current_ch} -> {next_ch} due to Jamming.")
            
        self.last_step_jammed = is_jammed

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
        self.update_channel_logic()

        if self.is_hovering:
            self.hover_timer += self.dt
            if self.hover_timer >= self.hover_duration:
                # Finished hovering, move to next node
                self.is_hovering = False
                self.hover_timer = 0.0
                
                # Get next target from queue
                if self.waypoint_queue:
                    self.target_node = self.waypoint_queue.pop(0)
                else:
                     # If queue empty, recalculate path or restart?
                     # Let's restart the tour from current position to cover dynamic changes if any, 
                     # or just loop. For now, simple loop: recalculate full path from current pos.
                     self._calculate_nn_path()
                     if self.waypoint_queue:
                         self.target_node = self.waypoint_queue.pop(0)
            else:
                # Stay hovering
                return np.zeros(2, dtype=np.float32)

        # Navigation Mode
        if self.target_node is None:
             if self.env.nodes:
                 self._calculate_nn_path()
                 if self.waypoint_queue:
                     self.target_node = self.waypoint_queue.pop(0)
             else:
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
        self.hover_timer = 0.0
        self.is_hovering = False
        self.waypoint_queue = []
        self.target_node = None
        # Path will be calculated on first get_action call or explicitly here if env is ready
        if self.env.nodes and self.env.uav:
            self._calculate_nn_path()
            if self.waypoint_queue:
                self.target_node = self.waypoint_queue.pop(0)

    def _calculate_nn_path(self):
        """
        Calculates Traveling Salesman Path using Nearest Neighbor Heuristic.
        Starting from current UAV position.
        """
        if not self.env.nodes or not self.env.uav:
            return

        # Clone list to not modify original
        unvisited = self.env.nodes[:]
        current_pos = np.array([self.env.uav.x, self.env.uav.y])
        
        path = []
        
        while unvisited:
            # Find closest node to current_pos
            closest_node = None
            min_dist = float('inf')
            closest_idx = -1
            
            for i, node in enumerate(unvisited):
                dist = np.linalg.norm(np.array([node.x, node.y]) - current_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_node = node
                    closest_idx = i
            
            # Add to path
            path.append(closest_node)
            
            # Update current_pos and remove from unvisited
            current_pos = np.array([closest_node.x, closest_node.y])
            unvisited.pop(closest_idx)
            
        self.waypoint_queue = path
