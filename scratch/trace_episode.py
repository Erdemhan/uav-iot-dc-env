import sys
import os
sys.path.append(r"e:\Projeler\TEZ\uav-iot-dc-env")

import numpy as np
from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.config import UAVConfig
from confs.env_config import EnvConfig

# Override PERSISTENCE_THRESHOLD to 1
UAVConfig.PERSISTENCE_THRESHOLD = 1

print("--- Running Trace Episode with PERSISTENCE_THRESHOLD = 1 ---")
env = UAV_IoT_PZ_Env(logger=None, auto_uav=True)
obs, infos = env.reset(seed=106)

print("Initial UAV Channels:", [u.current_channel for u in env.uavs])

# Let's run a full episode and trace
for step in range(1, 101):
    # Jammer action: always choose channel 0, max power level
    jam_ch = 0
    jam_p = 9
    actions = {'jammer_0': jam_ch * 10 + jam_p if env.flatten_actions else np.array([jam_ch, jam_p])}
    
    for agent in env.agents:
        if "node" in agent:
            actions[agent] = 0
            
    obs, rewards, terminations, truncations, infos = env.step(actions)
    
    statuses = [node.connection_status for node in env.nodes]
    uav_channels = [u.current_channel for u in env.uavs]
    jammed_cnt = statuses.count(2)
    out_range_cnt = statuses.count(1)
    connected_cnt = statuses.count(0)
    
    if jammed_cnt > 0 or any(ch != 0 for ch in uav_channels) or step in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
        print(f"Step {step:3d}: UAV Channels: {uav_channels} | Connected: {connected_cnt:2d}, OutRange: {out_range_cnt:2d}, Jammed: {jammed_cnt:2d}")
        cnts = [ctrl.jammed_step_counter for ctrl in env.uav_controllers]
        print(f"          Jam Counters: {cnts}")
        
    if all(terminations.values()) or all(truncations.values()):
        break

print("Trace Complete.")
