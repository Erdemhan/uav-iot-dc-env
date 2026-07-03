# tune_reward.py
# Phase 2: Joint Reward Optimization
# ---
# Runs AFTER tune_models.py has completed Phase 1 for PPO, DQN and QJC.
# Searches for (W_SUCCESS, W_COST) that maximizes the MEAN JSR across all 3 algorithms.
# Results are saved to: artifacts/tune_reward_phase2_<timestamp>/
# Best weights are written to: confs/tuned_configs.json["reward"]
#
# Usage:
#   python scripts/tune_reward.py --num-samples 20 --iterations 500 --num-workers 14 --use-gpu True

import os
import sys
import argparse
import warnings
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="pettingzoo")

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.env_config import EnvConfig
from confs.model_config import GlobalConfig
from confs.opt_config import OptConfig


SEEDS = range(100, 130)  # Same 30 seeds as Phase 1


# ---------------------------------------------------------------------------
# Env Creator (required for RLlib registration)
# ---------------------------------------------------------------------------
def env_creator(config):
    return ParallelPettingZooEnv(
        UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
    )


# ---------------------------------------------------------------------------
# 30-Seed Evaluation (copied from tune_models.py for independence)
# ---------------------------------------------------------------------------
def run_30seeds_eval(algo_agent, algo_name, env_config, lstm_cell_size=256,
                     q_table=None, q_counts=None):
    ep_jsrs = []
    w_success  = env_config.get("W_SUCCESS",  EnvConfig.W_SUCCESS)
    w_tracking = 1.0 - w_success
    w_cost     = env_config.get("W_COST",     EnvConfig.W_COST)

    for seed in SEEDS:
        eval_env = UAV_IoT_PZ_Env(logger=None, auto_uav=True,
                                   flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        eval_env.w_success  = w_success
        eval_env.w_tracking = w_tracking
        eval_env.w_cost     = w_cost
        obs, _ = eval_env.reset(seed=seed)

        if algo_name == "Baseline" and q_table is not None:
            eval_env.attacker.q_table        = q_table.copy()
            eval_env.attacker.channel_counts = q_counts.copy()
            eval_env.attacker.temp_xi        = 0.0  # greedy

        lstm_state = (
            [np.zeros(lstm_cell_size, dtype=np.float32),
             np.zeros(lstm_cell_size, dtype=np.float32)]
            if algo_name == "PPO-LSTM" else []
        )

        terminated = False
        steps = ep_jammed = ep_reachable = 0

        while not terminated and steps < EnvConfig.MAX_STEPS:
            actions = {}
            steps += 1

            if algo_name == "Baseline":
                from confs.model_config import QJCConfig
                jam_ch = eval_env.attacker.select_channel_qjc()
                jam_p  = QJCConfig.MAX_POWER_LEVEL
                jam_action = (jam_ch * 10 + jam_p if GlobalConfig.FLATTEN_ACTIONS
                              else np.array([jam_ch, jam_p]))
            elif algo_agent and "jammer_0" in obs:
                try:
                    if algo_name == "PPO-LSTM":
                        res = algo_agent.compute_single_action(
                            obs["jammer_0"], state=lstm_state, policy_id="jammer_policy", explore=False
                        )
                        if isinstance(res, tuple):
                            jam_action = res[0]
                            lstm_state = res[1]
                        else:
                            jam_action = res
                    else:
                        jam_action = algo_agent.compute_single_action(
                            obs["jammer_0"], policy_id="jammer_policy", explore=False
                        )
                except Exception:
                    jam_action = 0
            else:
                jam_action = 0


            actions["jammer_0"] = jam_action
            for ag in eval_env.agents:
                if "node" in ag:
                    actions[ag] = 0

            obs, rewards, term, trunc, infos = eval_env.step(actions)
            reachable = sum(1 for n in eval_env.nodes if n.connection_status != 1)
            jammed    = infos.get("jammer_0", {}).get("jammed_count", 0)
            ep_jammed   += jammed
            ep_reachable += reachable if reachable > 0 else 0
            terminated = any(term.values()) or any(trunc.values())

        jsr = (ep_jammed / ep_reachable * 100.0) if ep_reachable > 0 else 0.0
        ep_jsrs.append(jsr)

    return float(np.mean(ep_jsrs))


# ---------------------------------------------------------------------------
# Helper: Build, train and evaluate one RLlib algorithm
# ---------------------------------------------------------------------------
def _run_rllib_algo(algo_name, model_params, env_config_override,
                    iterations, num_workers, num_gpus):
    register_env("uav_iot_reward_v1", env_creator)

    dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
    obs_space = dummy_env.observation_space("jammer_0")
    act_space = dummy_env.action_space("jammer_0")
    node_obs  = dummy_env.observation_space("node_0")
    node_act  = dummy_env.action_space("node_0")

    architecture = model_params.get("architecture", None)
    if architecture is not None:
        fcnet_hiddens = list(architecture)
    else:
        fcnet_hiddens = [model_params.get("layer_size", 256)] * model_params.get("num_layers", 2)

    lr    = model_params.get("lr", 5e-4)
    gamma = model_params.get("gamma", 0.97)

    if algo_name == "PPO":
        from ray.rllib.algorithms.ppo import PPOConfig
        cfg = PPOConfig()
        model_cfg = {"fcnet_hiddens": fcnet_hiddens}
    elif algo_name == "PPO-LSTM":
        from ray.rllib.algorithms.ppo import PPOConfig
        cfg = PPOConfig()
        model_cfg = {
            "fcnet_hiddens": fcnet_hiddens,
            "use_lstm": True,
            "lstm_cell_size": model_params.get("lstm_cell_size", 256),
            "max_seq_len": model_params.get("max_seq_len", 20)
        }
    elif algo_name == "DQN":
        from ray.rllib.algorithms.dqn import DQNConfig
        cfg = DQNConfig()
        cfg.training(
            target_network_update_freq=model_params.get("target_network_update_freq", 500),
            double_q=True, dueling=True,
            replay_buffer_config={"type": "ReplayBuffer", "capacity": 50000},
            num_steps_sampled_before_learning_starts=0
        )
        model_cfg = {"fcnet_hiddens": fcnet_hiddens}
    else:
        raise ValueError(f"Unknown algo: {algo_name}")


    cfg = (
        cfg
        .environment("uav_iot_reward_v1", env_config={"seed": GlobalConfig.RANDOM_SEED})
        .framework("torch")
        .debugging(seed=GlobalConfig.RANDOM_SEED, log_level="WARN")
        .env_runners(num_env_runners=num_workers, rollout_fragment_length=100)  # 100 steps = 1 full episode (MAX_STEPS=100)
        .training(model=model_cfg, train_batch_size=1000, lr=lr, gamma=gamma)
        .multi_agent(
            policies={
                "jammer_policy": (None, obs_space, act_space, {}),
                "node_policy":   (None, node_obs,  node_act,  {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs:
                "jammer_policy" if agent_id == "jammer_0" else "node_policy",
            policies_to_train=["jammer_policy"],
        )
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=num_gpus)
    )

    algo = cfg.build()
    for _ in range(iterations):
        algo.train()
    jsr = run_30seeds_eval(
        algo_agent=algo, algo_name=algo_name, env_config=env_config_override,
        lstm_cell_size=model_params.get("lstm_cell_size", 256) if algo_name == "PPO-LSTM" else 256
    )
    algo.stop()
    return jsr



# ---------------------------------------------------------------------------
# Helper: Train QJC and evaluate
# ---------------------------------------------------------------------------
def _run_qjc_algo(qjc_params, env_config_override, iterations):
    tau_0     = qjc_params.get("tau_0",     1e-4)
    gamma_q   = qjc_params.get("gamma",     0.97)
    temp_xi   = qjc_params.get("temp_xi",   5.0)
    mu_offset = qjc_params.get("mu_offset", 1.5)

    total_eps = iterations * 10  # 1 iter = 10 episodes for QJC
    env = UAV_IoT_PZ_Env(logger=None, auto_uav=True)
    learned_q = learned_counts = None

    for ep in range(1, total_eps + 1):
        obs, _ = env.reset()
        if learned_q is not None:
            env.attacker.q_table        = learned_q
            env.attacker.channel_counts = learned_counts
        env.attacker.tau_0     = tau_0
        env.attacker.gamma     = gamma_q
        env.attacker.temp_xi   = temp_xi
        env.attacker.mu_offset = mu_offset

        done = False
        while not done:
            ch = env.attacker.select_channel_qjc()
            actions = {"jammer_0": np.array([ch, 9])}
            for ag in env.agents:
                if "node" in ag:
                    actions[ag] = 0
            obs, rewards, term, trunc, infos = env.step(actions)
            env.attacker.update_qjc(env.attacker.current_channel, rewards["jammer_0"])
            done = any(term.values()) or any(trunc.values())

        learned_q      = env.attacker.q_table.copy()
        learned_counts = env.attacker.channel_counts.copy()

    return run_30seeds_eval(
        algo_agent=None, algo_name="Baseline",
        env_config=env_config_override,
        q_table=learned_q, q_counts=learned_counts
    )


# ---------------------------------------------------------------------------
# Ray Tune Trainable — Phase 2 Joint Trial
# ---------------------------------------------------------------------------
def train_reward_trial(config):
    """Evaluate (W_SUCCESS, W_COST) pair on PPO + DQN + QJC.
    Objective = mean JSR across all 3 algorithms."""
    w_success  = config["W_SUCCESS"]
    w_cost     = config["W_COST"]
    w_tracking = 1.0 - w_success
    iterations  = config["iterations"]
    num_workers = config["num_workers"]
    num_gpus    = config["num_gpus"]

    env_cfg = {"W_SUCCESS": w_success, "W_TRACKING": w_tracking, "W_COST": w_cost}

    # 1. QJC
    jsr_qjc = _run_qjc_algo(config["qjc_params"], env_cfg, iterations)

    # 2. PPO
    jsr_ppo = _run_rllib_algo("PPO", config["ppo_params"], env_cfg,
                               iterations, num_workers, num_gpus)

    # 3. DQN
    jsr_dqn = _run_rllib_algo("DQN", config["dqn_params"], env_cfg,
                               iterations, num_workers, num_gpus)

    # 4. PPO-LSTM
    jsr_ppo_lstm = _run_rllib_algo("PPO-LSTM", config["ppo_lstm_params"], env_cfg,
                                    iterations, num_workers, num_gpus)

    mean_jsr = float(np.mean([jsr_qjc, jsr_ppo, jsr_dqn, jsr_ppo_lstm]))

    tune.report({
        "objective":    mean_jsr,
        "jsr":          mean_jsr,
        "jsr_qjc":      jsr_qjc,
        "jsr_ppo":      jsr_ppo,
        "jsr_dqn":      jsr_dqn,
        "jsr_ppo_lstm": jsr_ppo_lstm,
        "W_SUCCESS":    w_success,
        "W_COST":       w_cost,
    })



# ---------------------------------------------------------------------------
# Optuna visualization (same as tune_models.py)
# ---------------------------------------------------------------------------
def save_optuna_visualizations(study, optuna_dir):
    import optuna.visualization.matplotlib as vis_mpl
    os.makedirs(optuna_dir, exist_ok=True)
    for name, fn in [
        ("optimization_history", vis_mpl.plot_optimization_history),
        ("param_importances",    vis_mpl.plot_param_importances),
    ]:
        try:
            fn(study)
            plt.tight_layout()
            plt.savefig(os.path.join(optuna_dir, f"{name}.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not plot {name}: {e}")
            plt.close()


def short_trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: Joint Reward Optimization — finds (W_SUCCESS, W_COST) that "
            "maximizes mean JSR across PPO, DQN and QJC.\n"
            "Requires Phase 1 (tune_models.py) to be completed for all 3 algorithms first."
        )
    )
    parser.add_argument("--num-samples",  type=int,  default=20,
                        help="Number of (W_SUCCESS, W_COST) combinations to try")
    parser.add_argument("--iterations",   type=int,  default=500,
                        help="Training iterations per algorithm per trial")
    parser.add_argument("--num-workers",  type=int,  default=10,
                        help="Env runners per trial. 10 × 100 steps = 1000 (train_batch_size), STRICT_PACK on 11 CPUs per machine.")
    parser.add_argument("--use-gpu",      type=bool, default=True,
                        help="Use GPU for PPO/DQN training")
    args = parser.parse_args()

    if args.use_gpu and not torch.cuda.is_available():
        print("[WARN] GPU requested but CUDA unavailable. Falling back to CPU.")
        args.use_gpu = False

    # -- Load Phase 1 tuned configs --
    tuned_cfg_path = os.path.join(PROJECT_ROOT, "confs", "tuned_configs.json")
    if not os.path.exists(tuned_cfg_path):
        print(f"[ERROR] {tuned_cfg_path} not found.")
        print("        Run tune_models.py for PPO, DQN and QJC first.")
        raise SystemExit(1)

    with open(tuned_cfg_path, "r", encoding="utf-8") as f:
        tuned_configs = json.load(f)

    missing = [k for k in ["ppo", "dqn", "ppo_lstm", "qjc"] if k not in tuned_configs]
    if missing:
        print(f"[ERROR] Missing Phase 1 results for: {missing}")
        print("        Complete Phase 1 for each algorithm before running Phase 2.")
        raise SystemExit(1)

    print("[OK] Phase 1 configs loaded:")
    for k in ["ppo", "dqn", "ppo_lstm", "qjc"]:
        print(f"     {k.upper()}: {tuned_configs[k]}")


    # -- Connect to Ray --
    runtime_env = {
        "working_dir": ".",
        "excludes": ["**/logs", "**/.venv", "**/artifacts", "**/comparison",
                     "**/scratch", "**/.git", "**/*.png", "**/*.json"]
    }
    try:
        ray.init(address="auto", runtime_env=runtime_env)
        print("[OK] Connected to Ray cluster.")
    except Exception as e:
        print(f"[WARN] Ray auto-connect failed: {e}. Starting local instance.")
        ray.init(runtime_env=runtime_env)

    # -- Setup run dir --
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name  = f"tune_reward_phase2_{timestamp}"
    run_dir   = os.path.join(PROJECT_ROOT, "artifacts", run_name)
    os.makedirs(run_dir, exist_ok=True)

    metadata = {
        "timestamp":   timestamp,
        "type":        "phase2_joint_reward",
        "num_samples": args.num_samples,
        "iterations":  args.iterations,
        "num_workers": args.num_workers,
        "use_gpu":     args.use_gpu,
        "phase1_ppo":      tuned_configs["ppo"],
        "phase1_dqn":      tuned_configs["dqn"],
        "phase1_ppo_lstm": tuned_configs["ppo_lstm"],
        "phase1_qjc":      tuned_configs["qjc"],
    }

    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # -- Search Space --
    search_space = {
        "W_SUCCESS": tune.uniform(OptConfig.REWARD_W_SUCCESS_MIN, OptConfig.REWARD_W_SUCCESS_MAX),
        "W_COST":    tune.loguniform(OptConfig.REWARD_W_COST_MIN, OptConfig.REWARD_W_COST_MAX),
    }

    trial_constant_config = {
        "iterations":  args.iterations,
        "num_workers": args.num_workers,
        "num_gpus":    1 if args.use_gpu else 0,
        "ppo_params":      tuned_configs["ppo"],
        "dqn_params":      tuned_configs["dqn"],
        "ppo_lstm_params": tuned_configs["ppo_lstm"],
        "qjc_params":      tuned_configs["qjc"],
    }
    full_config = {**trial_constant_config, **search_space}


    # -- Resources --
    from ray.tune import PlacementGroupFactory
    # STRICT_PACK: all bundles (1 learner + 10 env runners = 11 CPUs) on the same machine.
    # Remaining 11 CPUs per machine are idle/isolated — no cross-trial interference.
    bundles = ([{"CPU": 1, "GPU": 1 if args.use_gpu else 0}]
               + [{"CPU": 1}] * args.num_workers)
    trial_resources = PlacementGroupFactory(bundles, strategy="STRICT_PACK")

    # -- Search Algorithm --
    optuna_search = OptunaSearch(metric="objective", mode="max")

    opt_local_dir = os.path.join(run_dir, "tune_results")

    print(f"\n==================================================")
    print(f" Phase 2 JOINT Reward Optimization")
    print(f" Algorithms:  PPO + DQN + PPO-LSTM + QJC | Objective: mean JSR")
    print(f" Trials:      {args.num_samples} | Iterations/algo: {args.iterations}")
    print(f" Output:      {run_dir}")
    print(f"==================================================\n")


    analysis = tune.run(
        train_reward_trial,
        config=full_config,
        resources_per_trial=trial_resources,
        search_alg=optuna_search,
        scheduler=None,  # No early stopping — every trial runs fully
        num_samples=args.num_samples,
        storage_path=opt_local_dir,
        name="optuna_study",
        trial_dirname_creator=short_trial_dirname_creator,
        verbose=1
    )

    # -- Extract and save results --
    study = (optuna_search.study if hasattr(optuna_search, "study")
             else optuna_search._ot_study)
    optuna_dir = os.path.join(run_dir, "optuna")
    os.makedirs(optuna_dir, exist_ok=True)

    best_trial = study.best_trial
    print(f"\n==================================================")
    print(f" Phase 2 Optimization Completed!")
    print(f" Best Trial #{best_trial.number} | Mean JSR: {best_trial.value:.4f}%")
    print(f" W_SUCCESS = {best_trial.params['W_SUCCESS']:.4f}")
    print(f" W_COST    = {best_trial.params['W_COST']:.6f}")
    print(f"==================================================\n")

    # Save best_params.json
    best_results = {
        "best_trial_number": best_trial.number,
        "best_value":        best_trial.value,
        "params":            best_trial.params,
    }
    with open(os.path.join(optuna_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=4)

    # Save to tuned_configs.json["reward"]
    os.makedirs(os.path.dirname(tuned_cfg_path), exist_ok=True)
    current_tuned = {}
    if os.path.exists(tuned_cfg_path):
        try:
            with open(tuned_cfg_path, "r", encoding="utf-8") as f:
                current_tuned = json.load(f)
        except Exception:
            pass
    current_tuned["reward"] = best_trial.params
    with open(tuned_cfg_path, "w", encoding="utf-8") as f:
        json.dump(current_tuned, f, indent=4)
    print(f"[OK] Best reward weights saved to: {tuned_cfg_path}")

    # Save all trials with per-algo JSR breakdown
    trials_data = []
    for trial in study.trials:
        if trial.state.is_finished():
            dur = 0.0
            if trial.datetime_complete and trial.datetime_start:
                dur = (trial.datetime_complete - trial.datetime_start).total_seconds()
            trials_data.append({
                "number":           trial.number,
                "value":            trial.value,
                "state":            str(trial.state),
                "params":           trial.params,
                "duration_seconds": dur,
            })
    with open(os.path.join(optuna_dir, "optuna_trials.json"), "w", encoding="utf-8") as f:
        json.dump(trials_data, f, indent=4)

    save_optuna_visualizations(study, optuna_dir)
    print(f"[OK] Visualizations saved in: {optuna_dir}")

    # Write active reward run file for dashboard
    reward_run_file = os.path.join(PROJECT_ROOT, "dashboard_active_reward_run.txt")
    with open(reward_run_file, "w", encoding="utf-8") as f:
        f.write(run_dir)
    print(f"[OK] Dashboard active reward run set to: {run_dir}")
    print(f"     Open: http://localhost:5000/reward_opt.html")


if __name__ == "__main__":
    main()
