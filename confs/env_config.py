
class EnvConfig:
    """
    Simulation Environment and Scenario Parameters.
    Separated from physical/system parameters.
    """
    # Scenario
    NUM_NODES = 5
    AREA_SIZE = 1000.0 # m
    
    # Attacker
    ATTACKER_POS_X = 600.0 # Absolute or relative logic can be used in env
    ATTACKER_POS_Y = 600.0 
    MAX_JAMMING_POWER = 0.1 # Watts (Previously 0.5 in action space)

    # Transmission Power (Scenario specific)
    P_TX_NODE = 0.002 # Watt (Reduced to see Out of Range)
    P_TX_UAV = 0.05  # Watt

    # Simulation Stepping
    MAX_STEPS = 100
    STEP_TIME = 5.0 # Seconds (dt)

    # Initial UAV State & Dynamics
    UAV_START_X = 500.0 # Center
    UAV_START_Y = 500.0 # Center
    UAV_START_Z = 100.0 # H (This duplicates UAVConfig.H slightly but is init pos)
    
    UAV_RADIUS = 225.0 # Circular path radius
    UAV_SPEED = 5.0 # m/s
    UAV_START_ANGLE = 15.0 # degrees or radians (Env uses radians inside usually, but lets say init val)
    
    # Note: UAVConfig.H is physical operating altitude. UAV_START_Z should match or be set, 
    # but strictly speaking H is a system parameter. We can reference UAVConfig.H in env.
