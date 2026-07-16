"""
evaluate_reachable_norm.py - v2
Orijinal evaluate_paper_robustness.py uzerine insa edilmistir.
Sadece ic donguye ep_tracking_reachable sayaci eklenmistir.

Yeni metrik:
  Track_Reachable = ep_tracking_reachable / ep_reachable * 100
  Power_Gap       = Track_Reachable - JSR   (dogru kanalda ama jam yapamadi)
  Channel_Gap     = 100 - Track_Reachable   (yanlis kanalda hiç sansi yok)
"""

import os, sys, argparse, json
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.config import UAVConfig
from confs.env_config import EnvConfig
from confs.model_config import GlobalConfig, PPOLSTMConfig, QJCConfig
from confs.opt_config import OptConfig

SEEDS = OptConfig.EVAL_SEEDS

ALGOS = ["Baseline", "PPO", "DQN", "PPO-LSTM"]

def load_env_config_from_metadata(run_dir):
    """metadata.json'dan EnvConfig'i override et — training env ile esles."""
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"  [WARN] metadata.json bulunamadi: {meta_path}")
        return
    with open(meta_path) as f:
        meta = json.load(f)
    ec = meta.get("env_config", {})
    from confs.env_config import EnvConfig
    if "NUM_NODES"        in ec: EnvConfig.NUM_NODES        = ec["NUM_NODES"]
    if "NUM_UAVS"         in ec: EnvConfig.NUM_UAVS         = ec["NUM_UAVS"]
    if "AREA_SIZE"        in ec: EnvConfig.AREA_SIZE        = ec["AREA_SIZE"]
    if "ATTACKER_POS_X"   in ec: EnvConfig.ATTACKER_POS_X   = ec["ATTACKER_POS_X"]
    if "ATTACKER_POS_Y"   in ec: EnvConfig.ATTACKER_POS_Y   = ec["ATTACKER_POS_Y"]
    if "MAX_JAMMING_POWER" in ec: EnvConfig.MAX_JAMMING_POWER = ec["MAX_JAMMING_POWER"]
    if "P_TX_NODE"        in ec: EnvConfig.P_TX_NODE        = ec["P_TX_NODE"]
    if "P_TX_UAV"         in ec: EnvConfig.P_TX_UAV         = ec["P_TX_UAV"]
    if "MAX_STEPS"        in ec: EnvConfig.MAX_STEPS        = ec["MAX_STEPS"]
    if "STEP_TIME"        in ec: EnvConfig.STEP_TIME        = ec["STEP_TIME"]
    if "UAV_START_X"      in ec: EnvConfig.UAV_START_X      = ec["UAV_START_X"]
    if "UAV_START_Y"      in ec: EnvConfig.UAV_START_Y      = ec["UAV_START_Y"]
    if "UAV_START_Z"      in ec: EnvConfig.UAV_START_Z      = ec["UAV_START_Z"]
    if "UAV_SPEED"        in ec: EnvConfig.UAV_SPEED        = ec["UAV_SPEED"]
    if "W_SUCCESS"        in ec: EnvConfig.W_SUCCESS        = ec["W_SUCCESS"]
    if "W_TRACKING"       in ec: EnvConfig.W_TRACKING       = ec["W_TRACKING"]
    if "W_COST"           in ec: EnvConfig.W_COST           = ec["W_COST"]
    obs_dim = EnvConfig.get_obs_dim()
    print(f"  EnvConfig override: NUM_NODES={EnvConfig.NUM_NODES}, NUM_UAVS={EnvConfig.NUM_UAVS}, "
          f"AREA_SIZE={EnvConfig.AREA_SIZE}, obs_dim={obs_dim}")


def env_creator(config):
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

def find_latest_checkpoint(base_dir):
    import glob
    search_pattern = os.path.join(base_dir, "**", "checkpoint_*")
    ckpt_dirs = glob.glob(search_pattern, recursive=True)
    ckpt_dirs = [os.path.abspath(d) for d in ckpt_dirs if os.path.isdir(d)]
    if not ckpt_dirs:
        return None
    return max(ckpt_dirs, key=os.path.getmtime)

def evaluate_algo(algo_name, run_dir):
    print(f"\n--- {algo_name} ---")

    algo_agent = None
    lstm_cell_size = 0

    if algo_name in ["PPO", "DQN", "PPO-LSTM"]:
        algo_dir = os.path.join(run_dir, algo_name.lower().replace("-", "_"))
        ckpt = find_latest_checkpoint(algo_dir)
        if not ckpt:
            print(f"  Checkpoint bulunamadi: {algo_dir}")
            return None
        try:
            algo_agent = Algorithm.from_checkpoint(ckpt)
        except Exception as e:
            print(f"  Yukleme hatasi: {e}")
            return None
        if algo_name == "PPO-LSTM":
            lstm_cell_size = PPOLSTMConfig.LSTM_CELL_SIZE

    results = {
        "JSR": [],
        "Track_AllSteps": [],
        "Track_Reachable": [],
        "Power_Gap": [],
        "Channel_Gap": [],
        "Reachable_Ratio": [],
    }

    for seed in SEEDS:
        env = UAV_IoT_PZ_Env(logger=None, auto_uav=True,
                             flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        obs, infos = env.reset(seed=seed)

        lstm_state = ([np.zeros(lstm_cell_size, dtype=np.float32),
                       np.zeros(lstm_cell_size, dtype=np.float32)]
                      if algo_name == "PPO-LSTM" else [])

        if algo_name == "Baseline":
            env.attacker.load_model(os.path.join(run_dir, "baseline"))
            env.attacker.temp_xi = 0
            env.attacker.channel_counts = np.zeros(env.attacker.num_channels) # Reset counts for online adaptation

        terminated = False
        steps = 0
        ep_jammed = 0
        ep_reachable = 0
        ep_tracking_all = 0
        ep_tracking_reachable = 0

        while not terminated and steps < EnvConfig.MAX_STEPS:
            steps += 1
            actions = {}

            if algo_name == "Baseline":
                jam_ch = env.attacker.select_channel_qjc()
                jam_p = QJCConfig.MAX_POWER_LEVEL
                actions["jammer_0"] = jam_ch * 10 + jam_p if GlobalConfig.FLATTEN_ACTIONS else np.array([jam_ch, jam_p])
            elif algo_agent and "jammer_0" in obs:
                try:
                    if algo_name == "PPO-LSTM":
                        res = algo_agent.compute_single_action(obs["jammer_0"], state=lstm_state,
                                                               policy_id="jammer_policy", explore=False)
                        actions["jammer_0"] = res[0] if isinstance(res, tuple) else res
                        if isinstance(res, tuple):
                            lstm_state = res[1]
                    else:
                        actions["jammer_0"] = algo_agent.compute_single_action(obs["jammer_0"],
                                                                                policy_id="jammer_policy", explore=False)
                except Exception as e:
                    if seed == list(SEEDS)[0]:  # sadece ilk seed'de logla
                        print(f"  [WARN] compute_single_action hatasi: {e}")
                    actions["jammer_0"] = 0
            else:
                actions["jammer_0"] = 0

            for ag in env.agents:
                if "node" in ag:
                    actions[ag] = 0

            obs, rewards, terms, truncs, infos = env.step(actions)
            terminated = any(terms.values()) or any(truncs.values())

            if algo_name == "Baseline":
                r_jam = rewards.get("jammer_0", 0)
                env.attacker.update_qjc(jam_ch, r_jam)

            # --- JSR ---
            reachable_count = sum(1 for n in env.nodes if n.connection_status != 1)
            jammed_c = infos["jammer_0"]["jammed_count"]
            ep_jammed += jammed_c
            if reachable_count > 0:
                ep_reachable += reachable_count

            # --- Tracking (iki versiyon) ---
            closest_uav = min(env.uavs,
                              key=lambda uav: np.linalg.norm(uav.position - env.attacker.position))
            ch_match = (env.attacker.current_channel == closest_uav.current_channel)
            if ch_match:
                ep_tracking_all += 1
                if reachable_count > 0:
                    ep_tracking_reachable += 1

        val_jsr       = (ep_jammed / ep_reachable * 100)            if ep_reachable > 0 else 0.0
        val_trk_all   = (ep_tracking_all / steps * 100)             if steps > 0 else 0.0
        val_trk_reach = (ep_tracking_reachable / ep_reachable * 100) if ep_reachable > 0 else 0.0
        val_pwr_gap   = val_trk_reach - val_jsr
        val_chn_gap   = 100.0 - val_trk_reach
        val_rch_ratio = (ep_reachable / steps * 100)                if steps > 0 else 0.0

        results["JSR"].append(val_jsr)
        results["Track_AllSteps"].append(val_trk_all)
        results["Track_Reachable"].append(val_trk_reach)
        results["Power_Gap"].append(val_pwr_gap)
        results["Channel_Gap"].append(val_chn_gap)
        results["Reachable_Ratio"].append(val_rch_ratio)

    return results


def print_summary(algo_name, res):
    jsr     = np.array(res["JSR"])
    trk_all = np.array(res["Track_AllSteps"])
    trk_rch = np.array(res["Track_Reachable"])
    pwr_gap = np.array(res["Power_Gap"])
    chn_gap = np.array(res["Channel_Gap"])
    rch_rat = np.array(res["Reachable_Ratio"])

    print(f"\n{'='*62}")
    print(f"  {algo_name}")
    print(f"{'='*62}")
    print(f"  Reachable adim orani : {rch_rat.mean():.1f}% +/- {rch_rat.std():.1f}%")
    print(f"  JSR                  : {jsr.mean():.1f}% +/- {jsr.std():.1f}%")
    print(f"  Track (all steps)    : {trk_all.mean():.1f}% +/- {trk_all.std():.1f}%   [eski metrik]")
    print(f"  Track (reachable)    : {trk_rch.mean():.1f}% +/- {trk_rch.std():.1f}%   [normalize]")
    print(f"  ----------------------------------------------------------")
    print(f"  Guc kaybi            : {pwr_gap.mean():.1f}% +/- {pwr_gap.std():.1f}%")
    print(f"    (Dogru kanalda ama jam edemedi -- mesafe/guc yetersiz)")
    print(f"  Kanal kaybi          : {chn_gap.mean():.1f}% +/- {chn_gap.std():.1f}%")
    print(f"    (Yanlis kanalda -- hic sansi yok)")
    print(f"  ----------------------------------------------------------")
    print(f"  KAYIP DOKUMU: Kanal={chn_gap.mean():.1f}%  Guc={pwr_gap.mean():.1f}%  Basari={jsr.mean():.1f}%  [toplam=100%]")


def main():
    from core.logger import setup_console_logging
    setup_console_logging("evaluate_reachable_norm")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to artifacts directory")
    args = parser.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    print(f"Run dir: {run_dir}")

    # --- Kritik: env_config'i training ile ayni hale getir ---
    load_env_config_from_metadata(run_dir)

    register_env("uav_iot_env", env_creator)
    register_env("uav_iot_ppo_v1", env_creator)
    register_env("uav_iot_dqn_v1", env_creator)
    register_env("uav_iot_ppo_lstm_v1", env_creator)
    ray.init(ignore_reinit_error=True)

    all_results = {}
    for algo in ALGOS:
        res = evaluate_algo(algo, run_dir)
        if res:
            all_results[algo] = res
            print_summary(algo, res)

    out_path = os.path.join(run_dir, "comparison", "reachable_norm_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nKaydedildi: {out_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
