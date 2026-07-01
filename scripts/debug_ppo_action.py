"""Debug: PPO checkpoint action computation"""
import os, sys, glob, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.model_config import GlobalConfig, QJCConfig
from confs.env_config import EnvConfig
import numpy as np

def env_creator(config):
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

register_env('uav_iot_ppo_v1', env_creator)
ray.init(ignore_reinit_error=True, logging_level='ERROR')

run_dir = os.path.abspath('artifacts/2026-06-25_10-11-44')
algo_dir = os.path.join(run_dir, 'ppo')
pattern = os.path.join(algo_dir, '**', 'checkpoint_*')
ckpt_dirs = [os.path.abspath(d) for d in glob.glob(pattern, recursive=True) if os.path.isdir(d)]
ckpt = max(ckpt_dirs, key=os.path.getmtime)
print(f"Loading checkpoint: {ckpt}")

algo = Algorithm.from_checkpoint(ckpt)
print("Checkpoint loaded OK")

# Policy map
try:
    policies = list(algo.workers.local_worker().policy_map.keys())
    print(f"Policy IDs: {policies}")
except Exception as e:
    print(f"local_worker error: {e}")
    # Alternative
    try:
        policies = list(algo.get_policy().keys()) if hasattr(algo.get_policy(), 'keys') else ["default"]
        print(f"Policy via get_policy: {policies}")
    except Exception as e2:
        print(f"get_policy error: {e2}")

env = UAV_IoT_PZ_Env(logger=None, auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
obs, infos = env.reset(seed=100)
print(f"jammer obs shape: {obs['jammer_0'].shape}")
print(f"jammer obs sample: {obs['jammer_0'][:5]}")

# Test jammer_policy
try:
    act = algo.compute_single_action(obs['jammer_0'], policy_id='jammer_policy', explore=False)
    print(f"Action (jammer_policy): {act}")
except Exception as e:
    print(f"jammer_policy ERROR: {type(e).__name__}: {e}")

# Run 5 steps and check actions
print("\n--- 5 step rollout ---")
for i in range(5):
    actions = {}
    try:
        actions['jammer_0'] = algo.compute_single_action(obs['jammer_0'], policy_id='jammer_policy', explore=False)
    except Exception as e:
        actions['jammer_0'] = 0
        print(f"  step {i}: action fallback, error={e}")

    for ag in env.agents:
        if 'node' in ag:
            actions[ag] = 0

    obs, rewards, terms, truncs, infos = env.step(actions)
    j_info = infos.get('jammer_0', {})
    jammed = j_info.get('jammed_count', -1)
    reachable = sum(1 for n in env.nodes if n.connection_status != 1)
    closest = min(env.uavs, key=lambda u: np.linalg.norm(u.position - env.attacker.position))
    ch_match = (env.attacker.current_channel == closest.current_channel)
    print(f"  step {i+1}: action={actions['jammer_0']}, jammer_ch={env.attacker.current_channel}, "
          f"uav_ch={closest.current_channel}, ch_match={ch_match}, jammed={jammed}, reachable={reachable}")

ray.shutdown()
