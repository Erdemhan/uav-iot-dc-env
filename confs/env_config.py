import numpy as np

class EnvConfig:
    """
    Simulation Environment and Scenario Parameters.
    Separated from physical/system parameters.
    """
    # Scenario
    NUM_NODES = 30
    NUM_UAVS = 2
    AREA_SIZE = 1000.0 # m
    
    # Attacker
    ATTACKER_POS_X = 500.0 # Absolute or relative logic can be used in env
    ATTACKER_POS_Y = 500.0 
    MAX_JAMMING_POWER = 1.0 # Watts (Industrial Standard: 30 dBm)

    # Transmission Power (Scenario specific)
    P_TX_NODE = 0.025  # Watt (14 dBm - Semtech SX1261 max efficiency)
    P_TX_UAV = 0.025   # Watt (14 dBm - Semtech SX1261 max efficiency)

    # Simulation Stepping
    MAX_STEPS = 100
    STEP_TIME = 5.0 # Seconds (dt)

    # Initial UAV State & Dynamics
    UAV_START_X = 250.0 # Center
    UAV_START_Y = 250.0 # Center
    UAV_START_Z = 100.0 # H (This duplicates UAVConfig.H slightly but is init pos)
    
    UAV_SPEED = 5.0 # m/s (Increased for larger area)

    # Reward weights
    W_SUCCESS = 0.8
    W_TRACKING = 0.2
    W_COST = 0.03

    @classmethod
    def get_obs_dim(cls):
        # 2 * NUM_UAVS (distances + channels) + 1 (jammer channel) + NUM_NODES (node rssis)
        return 2 * cls.NUM_UAVS + 1 + cls.NUM_NODES

    @staticmethod
    def build_state_vector(uav_distances, uav_channels, jammer_channel, node_rssis):
        return np.concatenate([
            np.array(uav_distances, dtype=np.float32),
            np.array(uav_channels, dtype=np.float32),
            np.array([jammer_channel], dtype=np.float32),
            np.array(node_rssis, dtype=np.float32)
        ])

    @staticmethod
    def parse_jammer_action(jam_action, flatten_actions):
        if flatten_actions:
            ch_idx = int(jam_action) // 10
            p_level = int(jam_action) % 10
        else:
            ch_idx = int(jam_action[0])
            p_level = int(jam_action[1])
        return ch_idx, p_level
