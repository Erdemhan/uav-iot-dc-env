# tune_models.py
import os
import sys
import argparse
import warnings
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# Suppress warnings and metrics logs
os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="pettingzoo")

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.algorithm import Algorithm

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.env_config import EnvConfig
from confs.model_config import GlobalConfig
from confs.opt_config import OptConfig


# Evaluation Constants
SEEDS = range(100, 130) # Seeds 100 to 129

def env_creator(config):
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

def run_30seeds_eval(algo_agent, algo_name, env_config, phase=1, lstm_cell_size=256, q_table=None, q_counts=None):
    """Evaluate current policy on 30 random seeds and calculate JSR, Tracking and Reward"""
    ep_rewards = []
    ep_jsrs = []
    ep_trackings = []
    ep_powers = []
    ep_sinrs = []
    
    # Save active EnvConfig weights to compute correct physical reward
    w_success = env_config.get("W_SUCCESS", EnvConfig.W_SUCCESS)
    w_tracking = 1.0 - w_success
    w_cost = env_config.get("W_COST", EnvConfig.W_COST)
    
    # Run over 30 seeds
    for seed in SEEDS:
        eval_env = UAV_IoT_PZ_Env(logger=None, auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        
        # Override weights inside evaluation env
        eval_env.w_success = w_success
        eval_env.w_tracking = w_tracking
        eval_env.w_cost = w_cost
        
        obs, infos = eval_env.reset(seed=seed)
        
        if algo_name == "Baseline":
            if q_table is not None:
                eval_env.attacker.q_table = q_table.copy()
                eval_env.attacker.channel_counts = q_counts.copy()
            eval_env.attacker.temp_xi = 0.0 # Greedy evaluation
            
        lstm_state = [np.zeros(lstm_cell_size, dtype=np.float32), np.zeros(lstm_cell_size, dtype=np.float32)] if algo_name == "PPO-LSTM" else []
        
        terminated = False
        steps = 0
        ep_jammed = 0
        ep_reachable = 0
        ep_tracking_reachable = 0
        ep_power_sum = 0
        ep_sinr_sum = 0
        ep_sinr_count = 0
        ep_reward_sum = 0
        
        while not terminated and steps < EnvConfig.MAX_STEPS:
            actions = {}
            steps += 1
            
            # Attacker Action
            jam_action = 0
            if algo_name == "Baseline":
                jam_ch = eval_env.attacker.select_channel_qjc()
                from confs.model_config import QJCConfig
                jam_p = QJCConfig.MAX_POWER_LEVEL
                jam_action = jam_ch * 10 + jam_p if GlobalConfig.FLATTEN_ACTIONS else np.array([jam_ch, jam_p])
            elif algo_agent:
                if "jammer_0" in obs:
                    jam_obs = obs["jammer_0"]
                    try:
                        if algo_name == "PPO-LSTM":
                            res = algo_agent.compute_single_action(jam_obs, state=lstm_state, policy_id="jammer_policy", explore=False)
                            if isinstance(res, tuple):
                                jam_action = res[0]
                                lstm_state = res[1]
                            else:
                                jam_action = res
                        else:
                            jam_action = algo_agent.compute_single_action(jam_obs, policy_id="jammer_policy", explore=False)
                    except:
                        jam_action = 0
                        
            actions["jammer_0"] = jam_action
            
            # Nodes Dummy
            for ag in eval_env.agents:
                if "node" in ag: actions[ag] = 0
                
            obs, rewards, term, trunc, infos = eval_env.step(actions)
            
            # Metrics accumulation
            r_jam = rewards.get("jammer_0", 0)
            ep_reward_sum += r_jam
            
            # --- Collect Stats ---
            reachable_count = sum(1 for n in eval_env.nodes if n.connection_status != 1)
            jammed_c = infos.get("jammer_0", {}).get("jammed_count", 0)
            
            ep_jammed += jammed_c
            if reachable_count > 0:
                ep_reachable += reachable_count
            
            # 2. Tracking
            closest_uav = min(eval_env.uavs, key=lambda uav: np.linalg.norm(uav.position - eval_env.attacker.position))
            ch_match = (eval_env.attacker.current_channel == closest_uav.current_channel)
            if ch_match:
                if reachable_count > 0:
                    ep_tracking_reachable += 1
            
            # 3. Power
            ep_power_sum += infos.get("jammer_0", {}).get("jammer_cost", 0)
            
            # 4. SINR
            step_sinr_sum = 0
            step_sinr_n = 0
            for k in range(EnvConfig.NUM_NODES):
                s = infos.get("jammer_0", {}).get(f"node_{k}_sinr", 0)
                step_sinr_sum += s
                step_sinr_n += 1
            
            if step_sinr_n > 0:
                step_avg_sinr = step_sinr_sum / step_sinr_n
                step_avg_sinr_db = 10 * np.log10(max(step_avg_sinr, 1e-12))
                ep_sinr_sum += step_avg_sinr_db
                ep_sinr_count += 1
            
            terminated = any(term.values()) or any(trunc.values())
            
        # Compile Seed stats
        jsr = (ep_jammed / ep_reachable * 100.0) if ep_reachable > 0 else 0.0
        track = (ep_tracking_reachable / ep_reachable * 100.0) if ep_reachable > 0 else 0.0
        power = ep_power_sum / steps if steps > 0 else 0.0
        sinr = ep_sinr_sum / ep_sinr_count if ep_sinr_count > 0 else 0.0
        
        ep_rewards.append(ep_reward_sum)
        ep_jsrs.append(jsr)
        ep_trackings.append(track)
        ep_powers.append(power)
        ep_sinrs.append(sinr)
        
    objective = float(np.mean(ep_rewards)) if phase == 1 else float(np.mean(ep_jsrs))
        
    return {
        "objective": objective,
        "reward": float(np.mean(ep_rewards)),
        "jsr": float(np.mean(ep_jsrs)),
        "tracking_acc": float(np.mean(ep_trackings)),
        "power": float(np.mean(ep_powers)),
        "sinr": float(np.mean(ep_sinrs))
    }

def train_rllib_trial(config):
    """Custom RLlib Tune Trainable supporting 30-seed robustness evaluations"""
    # Environment Setup
    register_env("uav_iot_ppo_v1", env_creator)
    
    algo_name = config["algo"]
    lr = config["lr"]
    gamma = config["gamma"]
    # Architecture is either a pre-defined list or legacy num_layers+layer_size
    architecture = config.get("architecture", None)
    if architecture is not None:
        fcnet_hiddens = list(architecture)
    else:
        # Legacy fallback
        num_layers = config.get("num_layers", 2)
        layer_size = config.get("layer_size", 256)
        fcnet_hiddens = [layer_size] * num_layers
    
    # Base Configuration building
    dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
    obs_space = dummy_env.observation_space("jammer_0")
    act_space = dummy_env.action_space("jammer_0")
    node_obs = dummy_env.observation_space("node_0")
    node_act = dummy_env.action_space("node_0")
    
    if algo_name == "PPO":
        from ray.rllib.algorithms.ppo import PPOConfig as RLlibPPOConfig
        cfg_obj = RLlibPPOConfig()
        model_cfg = {"fcnet_hiddens": fcnet_hiddens}
    elif algo_name == "PPO-LSTM":
        from ray.rllib.algorithms.ppo import PPOConfig as RLlibPPOConfig
        cfg_obj = RLlibPPOConfig()
        model_cfg = {
            "fcnet_hiddens": fcnet_hiddens,
            "use_lstm": True,
            "lstm_cell_size": config.get("lstm_cell_size", 256),
            "max_seq_len": config.get("max_seq_len", 20)
        }
    elif algo_name == "DQN":
        from ray.rllib.algorithms.dqn import DQNConfig as RLlibDQNConfig
        cfg_obj = RLlibDQNConfig()
        cfg_obj.training(
            target_network_update_freq=config.get("target_network_update_freq", 500),
            double_q=True,
            dueling=True,
            replay_buffer_config={"type": "ReplayBuffer", "capacity": 50000},
            num_steps_sampled_before_learning_starts=0
        )
        model_cfg = {"fcnet_hiddens": fcnet_hiddens}
        
    cfg_obj = (
        cfg_obj
        .environment("uav_iot_ppo_v1", env_config={"seed": GlobalConfig.RANDOM_SEED})
        .framework("torch")
        .debugging(seed=GlobalConfig.RANDOM_SEED)
        .env_runners(
            num_env_runners=config["num_workers"],
            rollout_fragment_length=100
        )
        .training(
            model=model_cfg,
            train_batch_size=1000,
            lr=lr,
            gamma=gamma,
        )
        .multi_agent(
            policies={
                "jammer_policy": (None, obs_space, act_space, {}),
                "node_policy": (None, node_obs, node_act, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: 
                "jammer_policy" if agent_id == "jammer_0" else "node_policy",
            policies_to_train=["jammer_policy"],
        )
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=config["num_gpus"])
        .debugging(log_level="WARN")
    )
    
    # Build Algorithm
    algo = cfg_obj.build()
    
    last_objective = -float('inf')
    last_jsr = 0.0
    last_track = 0.0
    last_power = 0.0
    last_sinr = 0.0
    
    iterations = config["iterations"]
    phase = config["phase"]
    
    for i in range(1, iterations + 1):
        algo.train()
        
        # Run 30 seeds evaluation every 5 iterations to save time
        if i % 5 == 0 or i == iterations:
            eval_metrics = run_30seeds_eval(
                algo_agent=algo, 
                algo_name=algo_name, 
                env_config=config["env_config"], 
                phase=phase,
                lstm_cell_size=config.get("lstm_cell_size", 256)
            )
            last_objective = eval_metrics["objective"]
            last_jsr = eval_metrics["jsr"]
            last_track = eval_metrics["tracking_acc"]
            last_power = eval_metrics["power"]
            last_sinr = eval_metrics["sinr"]
            
        tune.report({
            "training_iteration": i,
            "objective": last_objective,
            "jsr": last_jsr,
            "tracking_acc": last_track,
            "power": last_power,
            "sinr": last_sinr
        })
        
    algo.stop()

def train_qjc_trial(config):
    """Custom QJC Tabular Trainable supporting 30-seed robustness evaluations"""
    tau_0 = config["tau_0"]
    gamma = config["gamma"]
    temp_xi = config["temp_xi"]
    mu_offset = config["mu_offset"]
    iterations = config["iterations"]
    phase = config["phase"]
    
    # 1 iteration = 10 episodes for QJC (so iterations * 10 episodes total)
    total_episodes = iterations * 10
    
    env = UAV_IoT_PZ_Env(logger=None, auto_uav=True)
    learned_q_table = None
    learned_counts = None
    
    last_objective = -float('inf')
    last_jsr = 0.0
    last_track = 0.0
    last_power = 0.0
    last_sinr = 0.0
    
    for ep in range(1, total_episodes + 1):
        obs, infos = env.reset()
        if learned_q_table is not None:
            env.attacker.q_table = learned_q_table
            env.attacker.channel_counts = learned_counts
            
        # Override hyperparams dynamically
        env.attacker.tau_0 = tau_0
        env.attacker.gamma = gamma
        env.attacker.temp_xi = temp_xi
        env.attacker.mu_offset = mu_offset
        
        terminated = False
        truncations = False
        
        while not (terminated or truncations):
            actions = {}
            selected_channel = env.attacker.select_channel_qjc()
            selected_power_level = 9 # Max Power level
            
            actions['jammer_0'] = np.array([selected_channel, selected_power_level])
            for agent in env.agents:
                if "node" in agent: actions[agent] = 0
                
            obs, rewards, term, trunc, infos = env.step(actions)
            r_jam = rewards['jammer_0']
            ch_idx = env.attacker.current_channel
            env.attacker.update_qjc(ch_idx, r_jam)
            
            terminated = any(term.values())
            truncations = any(trunc.values())
            
        # Store counts
        learned_q_table = env.attacker.q_table.copy()
        learned_counts = env.attacker.channel_counts.copy()
        
        # Evaluate every 50 episodes (equivalent to 5 iterations)
        if ep % 50 == 0 or ep == total_episodes:
            eval_metrics = run_30seeds_eval(
                algo_agent=None,
                algo_name="Baseline",
                env_config=config["env_config"],
                phase=phase,
                q_table=learned_q_table,
                q_counts=learned_counts
            )
            last_objective = eval_metrics["objective"]
            last_jsr = eval_metrics["jsr"]
            last_track = eval_metrics["tracking_acc"]
            last_power = eval_metrics["power"]
            last_sinr = eval_metrics["sinr"]
            
        # Report simulation iterations (1 iteration = 10 episodes)
        if ep % 10 == 0:
            tune.report({
                "training_iteration": ep // 10,
                "objective": last_objective,
                "jsr": last_jsr,
                "tracking_acc": last_track,
                "power": last_power,
                "sinr": last_sinr
            })


def _study_with_str_arch(study):
    """Return a copy of the Optuna study with architecture lists converted to
    string representations so that parallel_coordinate and slice_plot can hash them."""
    import optuna
    import copy
    
    # Create a fresh in-memory study
    new_study = optuna.create_study(direction="maximize")
    for t in study.trials:
        if not t.state.is_finished():
            continue
        # Convert any list-valued param to its string representation
        new_params = {}
        for k, v in t.params.items():
            new_params[k] = str(v) if isinstance(v, list) else v
        frozen = optuna.trial.create_trial(
            params=new_params,
            distributions={
                k: (optuna.distributions.CategoricalDistribution([str(v) if isinstance(v, list) else v
                                                                  for v in d.choices])
                    if isinstance(d, optuna.distributions.CategoricalDistribution) and
                       any(isinstance(c, list) for c in d.choices)
                    else d)
                for k, d in t.distributions.items()
            },
            value=t.value
        )
        new_study.add_trial(frozen)
    return new_study

def save_optuna_visualizations(study, optuna_dir):
    """Draw and save Optuna study visualization plots using Matplotlib backend"""
    import optuna.visualization.matplotlib as vis_mpl
    os.makedirs(optuna_dir, exist_ok=True)
    
    # Study with architecture param as string (for hash-sensitive plots)
    study_str = _study_with_str_arch(study)
    
    # 1. Optimization History
    try:
        vis_mpl.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(optuna_dir, "optimization_history.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting optimization history: {e}")
        plt.close()
        
    # 2. Parameter Importance
    try:
        vis_mpl.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(optuna_dir, "param_importances.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting param importances: {e}")
        plt.close()
        
    # 3. Parallel Coordinate (uses str-arch study to avoid unhashable list)
    try:
        vis_mpl.plot_parallel_coordinate(study_str)
        plt.tight_layout()
        plt.savefig(os.path.join(optuna_dir, "parallel_coordinate.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting parallel coordinate: {e}")
        plt.close()
        
    # 4. Slice Plot (uses str-arch study to avoid unhashable list)
    try:
        vis_mpl.plot_slice(study_str)
        plt.tight_layout()
        plt.savefig(os.path.join(optuna_dir, "slice_plot.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting slice plot: {e}")
        plt.close()


def short_trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Per-Algorithm Model Hyperparameter Search (PPO, DQN, PPO-LSTM, QJC).\n"
                    "For Phase 2 (Joint Reward Optimization), use: python scripts/tune_reward.py"
    )
    parser.add_argument("--algo", type=str, default="PPO",
                        choices=["PPO", "DQN", "PPO-LSTM", "QJC"],
                        help="Algorithm to tune")
    parser.add_argument("--num-samples", type=int, default=30, help="Total number of trials to sample")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations per trial (each trial runs for this many algo.train() calls)")
    parser.add_argument("--num-workers",  type=int,  default=10,
                        help="Env runners per trial. 10 workers × 100 steps = train_batch_size=1000 exactly. 11 CPUs total (1 learner + 10 runners) per machine, STRICT_PACK.")
    parser.add_argument("--use-gpu", type=bool, default=True, help="Use RTX 3060 GPU for network updates")
    parser.add_argument("--max-concurrent", type=int, default=0, help="Maximum concurrent trials for this algorithm. 0 for unlimited.")
    args = parser.parse_args()

    
    # Auto-detect CUDA availability
    import torch
    if args.use_gpu and not torch.cuda.is_available():
        print("[WARN] GPU requested but CUDA is not available on this device. Falling back to CPU.")
        args.use_gpu = False
        
    # 1. Start or Connect to Ray Cluster
    print(f"Connecting to Ray Cluster...")
    runtime_env = {
        "working_dir": ".",
        "excludes": ["**/logs", "**/.venv", "**/artifacts", "**/comparison", "**/scratch", "**/.git", "**/*.png", "**/*.json"]
    }
    
    try:
        # Connect to existing cluster.
        # Head node is started with --num-cpus=0 --num-gpus=0, so Ray will
        # NOT schedule any trial trainable or env-runner on the head machine.
        # All compute work goes to worker nodes automatically.
        ray.init(address="auto", runtime_env=runtime_env)
        print("[OK] Connected to active Ray head node successfully!")
    except Exception as e:
        print(f"[WARN] Ray address='auto' connection failed: {e}. Starting local Ray instance...")
        # Fallback: single-machine mode (head becomes a worker too in this case)
        ray.init(runtime_env=runtime_env)
        
    # 2. Setup run directory and paths
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"tune_{args.algo.lower().replace('-', '_')}_phase{args.phase}_{timestamp}"
    run_dir = os.path.join(PROJECT_ROOT, "artifacts", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Write metadata to log folder
    metadata = {
        "timestamp": timestamp,
        "algo": args.algo,
        "phase": args.phase,
        "num_samples": args.num_samples,
        "iterations": args.iterations,
        "num_workers": args.num_workers,
        "use_gpu": args.use_gpu,
        "env_config": {
            "W_SUCCESS": EnvConfig.W_SUCCESS,
            "W_TRACKING": EnvConfig.W_TRACKING,
            "W_COST": EnvConfig.W_COST
        }
    }
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    # 3. Define Optuna Search Space and Search Alg
    optuna_search = OptunaSearch(metric="objective", mode="max")

    # Load existing tuned_configs (to avoid overwriting other algos)
    tuned_configs = {}
    tuned_cfg_path = os.path.join(PROJECT_ROOT, "confs", "tuned_configs.json")
    if os.path.exists(tuned_cfg_path):
        try:
            with open(tuned_cfg_path, "r", encoding="utf-8") as f:
                tuned_configs = json.load(f)
        except:
            pass

    # Phase 1: Model Hyperparameters
        if args.algo == "PPO":
            search_space = {
                "lr": tune.loguniform(OptConfig.RL_LR_MIN, OptConfig.RL_LR_MAX),
                "gamma": tune.uniform(OptConfig.RL_GAMMA_MIN, OptConfig.RL_GAMMA_MAX),
                "architecture": tune.choice(OptConfig.ARCH_CHOICES)
            }
        elif args.algo == "DQN":
            search_space = {
                "lr": tune.loguniform(OptConfig.RL_LR_MIN, OptConfig.RL_LR_MAX),
                "gamma": tune.uniform(OptConfig.RL_GAMMA_MIN, OptConfig.RL_GAMMA_MAX),
                "architecture": tune.choice(OptConfig.ARCH_CHOICES),
                "target_network_update_freq": tune.choice(OptConfig.DQN_TARGET_UPDATE_FREQ)
            }
        elif args.algo == "PPO-LSTM":
            search_space = {
                "lr": tune.loguniform(OptConfig.RL_LR_MIN, OptConfig.RL_LR_MAX),
                "gamma": tune.uniform(OptConfig.RL_GAMMA_MIN, OptConfig.RL_GAMMA_MAX),
                "architecture": tune.choice(OptConfig.ARCH_CHOICES),
                "lstm_cell_size": tune.choice(OptConfig.PPOLSTM_CELL_SIZE),
                "max_seq_len": tune.choice(OptConfig.PPOLSTM_MAX_SEQ_LEN)
            }

        elif args.algo == "QJC":
            search_space = {
                "tau_0": tune.loguniform(OptConfig.QJC_TAU_0_MIN, OptConfig.QJC_TAU_0_MAX),
                "gamma": tune.uniform(OptConfig.QJC_GAMMA_MIN, OptConfig.QJC_GAMMA_MAX),
                "temp_xi": tune.uniform(OptConfig.QJC_TEMP_XI_MIN, OptConfig.QJC_TEMP_XI_MAX),
                "mu_offset": tune.uniform(OptConfig.QJC_MU_OFFSET_MIN, OptConfig.QJC_MU_OFFSET_MAX)
            }

    # Phase 1 — Per-Algorithm Model Hyperparameter Search
    # ---------------------------------------------------------------------------

    opt_local_dir = os.path.join(run_dir, "tune_results")
    
    # Define trial resource allocations to prevent placement group conflicts
    if args.algo == "QJC":
        trial_resources = {
            "cpu": 1,
            "gpu": 0
        }
    else:
        from ray.tune import PlacementGroupFactory
        bundles = [{"CPU": 1, "GPU": 1 if args.use_gpu else 0}] + [{"CPU": 1}] * args.num_workers
        trial_resources = PlacementGroupFactory(bundles, strategy="STRICT_PACK")
    
    analysis = tune.run(
        trainable,
        config=full_config,
        resources_per_trial=trial_resources,
        search_alg=optuna_search,
        scheduler=scheduler,
        num_samples=args.num_samples,
        max_concurrent_trials=args.max_concurrent if args.max_concurrent > 0 else None,
        storage_path=opt_local_dir,
        name="optuna_study",
        trial_dirname_creator=short_trial_dirname_creator,
        verbose=1
    )

    
    # 5. Extract Optuna Study and Save plots/data
    study = optuna_search.study if hasattr(optuna_search, "study") else optuna_search._ot_study
    optuna_dir = os.path.join(run_dir, "optuna")
    os.makedirs(optuna_dir, exist_ok=True)
    
    # Save Best Parameters
    best_trial = study.best_trial
    print(f"\n==================================================")
    print(f" Optimization Completed!")
    print(f" Best Trial #{best_trial.number} | Score: {best_trial.value:.4f}")
    print(f" Best Parameters: {best_trial.params}")
    print(f"==================================================\n")
    
    best_results = {
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "params": best_trial.params
    }
    with open(os.path.join(optuna_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=4)
        
    # Save Phase 1 results to tuned_configs.json (for tune_reward.py Phase 2 use)
    os.makedirs(os.path.dirname(tuned_cfg_path), exist_ok=True)
    current_tuned = {}
    if os.path.exists(tuned_cfg_path):
        try:
            with open(tuned_cfg_path, "r", encoding="utf-8") as f:
                current_tuned = json.load(f)
        except:
            pass
    algo_key = args.algo.lower().replace("-", "_")
    current_tuned[algo_key] = best_trial.params
    with open(tuned_cfg_path, "w", encoding="utf-8") as f:
        json.dump(current_tuned, f, indent=4)
    print(f"[OK] Saved tuned configurations to: {tuned_cfg_path}")
    print(f"     When PPO, DQN and QJC are all done, run: python scripts/tune_reward.py")
        
    # Save all trials database
    trials_data = []
    for trial in study.trials:
        # Only log finished/pruned trials
        if trial.state.is_finished():
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "state": str(trial.state),
                "params": trial.params,
                "duration_seconds": (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete and trial.datetime_start else 0.0
            })
    with open(os.path.join(optuna_dir, "optuna_trials.json"), "w", encoding="utf-8") as f:
        json.dump(trials_data, f, indent=4)
        
    # Draw and Save Matplotlib Plots
    print(f"Drawing Optuna study visualization plots...")
    save_optuna_visualizations(study, optuna_dir)
    print(f"[OK] Visualizations and trial results saved successfully in: {optuna_dir}")
    
    # Expose this run directory as the active run for the Dashboard server
    # We write a file dashboard_active_run.txt in PROJECT_ROOT
    active_run_file = os.path.join(PROJECT_ROOT, "dashboard_active_run.txt")
    with open(active_run_file, "w", encoding="utf-8") as f:
        f.write(run_dir)
    print(f"Set dashboard active run to: {run_dir}")

if __name__ == "__main__":
    main()
