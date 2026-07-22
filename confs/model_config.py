
# --- Ray RLlib DQN Replay Buffer Bug Patch ---
try:
    import ray.rllib.algorithms.algorithm as rllib_algo
    original_create_buffer = rllib_algo.Algorithm._create_local_replay_buffer_if_necessary

    def patched_create_buffer(self, config, *args, **kwargs):
        has_patched = False
        original_type = None
        if "replay_buffer_config" in config and "type" in config["replay_buffer_config"]:
            original_type = config["replay_buffer_config"]["type"]
            if original_type is not None and not isinstance(original_type, str):
                has_patched = True
                if hasattr(original_type, "__name__"):
                    config["replay_buffer_config"]["type"] = original_type.__name__
                else:
                    config["replay_buffer_config"]["type"] = str(original_type)
        try:
            return original_create_buffer(self, config, *args, **kwargs)
        finally:
            if has_patched:
                config["replay_buffer_config"]["type"] = original_type

    rllib_algo.Algorithm._create_local_replay_buffer_if_necessary = patched_create_buffer
except Exception:
    pass
# ---------------------------------------------

class GlobalConfig:
    """
    Global configuration shared across all algorithms.
    Ensures reproducibility and consistency.
    """
    RANDOM_SEED = 42            # Random seed for reproducibility
    FLATTEN_ACTIONS = True      # Flatten MultiDiscrete to Discrete for DQN compatibility
    TRAIN_ITERATIONS = 10      # Number of training iterations (Quick test mode: 10)
    TRAIN_BATCH_SIZE = 1000     # Steps collected per iteration (PPO/DQN)
    
    # Early Stopping Parameters
    # EARLY_STOPPING_PATIENCE: Tüm zamanların en iyi (maksimum) ödülünden sonra 
    # kaç eğitim iterasyonu boyunca artış olmazsa eğitimin duracağını belirtir.
    # Not: 1 iterasyon = 1000 adım (10 epizot) -> 100 iterasyon = 100,000 adım (1000 epizot)
    EARLY_STOPPING_PATIENCE = 100
    
    # EARLY_STOPPING_MIN_REWARD: Sabır sayacının devreye girmesi için ajanın ulaşması gereken 
    # minimum ortalama ödül barajıdır (Bu baraj aşılana kadar erken durdurma tetiklenmez).
    # Önemli: Düşük iterasyonlu testler için erken durdurma barajını sıfıra yakın tutuyoruz.
    EARLY_STOPPING_MIN_REWARD = 0.0
    
    # Checkpointing Parameters
    CHECKPOINT_FREQ = 50                                     # Her N iterasyonda bir model yedekleme sıklığı
    KEEP_CHECKPOINTS_NUM = 3                                 # Disk üzerinde tutulacak en yüksek performanslı en iyi model sayısı
    CHECKPOINT_SCORE_ATTR = "env_runners/episode_reward_mean" # Checkpoint'leri sıralamak için kullanılan performans metriği

    # Shared Resource Parameters
    NUM_WORKERS = 10     # Number of parallel rollout workers
    USE_GPU = True       # GPU enabled or disabled (e.g. GTX 3080 with CUDA 12.1)

    

class QJCConfig:
    """
    Parameters for the Baseline QJC Algorithm (Liao et al. 2025).
    Used in simulation/entities.py and train_baseline.py
    """
    TAU_0     = 0.399    # Başlangıç sıcaklık sabiti  | HPO: tune_qjc_phase1_2026-07-21_11-36-04/optuna/best_params.json
    GAMMA     = 0.887    # İskonto faktörü             | HPO: tune_qjc_phase1_2026-07-21_11-36-04/optuna/best_params.json
    TEMP_XI   = 1.303    # Sıcaklık bozunma katsayısı | HPO: tune_qjc_phase1_2026-07-21_11-36-04/optuna/best_params.json
    MU_OFFSET = 1.475    # Kanal merkezi kayma ofseti | HPO: tune_qjc_phase1_2026-07-21_11-36-04/optuna/best_params.json
    TRAIN_EPISODES = GlobalConfig.TRAIN_ITERATIONS * 10  # Baseline episodes (200 episodes @ 100 steps each = 20k steps)
    SAVE_PATH = "baseline_q_table"          # Path to save Q-table
    MAX_POWER_LEVEL = 9                     # Maximum jammer power level (0-9)

class PPOConfig:
    """
    Parameters for Ray RLLib PPO Training.
    Used in train.py
    """

    # PPO Hyperparameters  |  Kaynak: tune_ppo_phase1_2026-07-17_17-50-43  |  Obj: 16.43
    LR    = 6.976e-4     # Öğrenme oranı   | HPO: tune_ppo_phase1_2026-07-17_17-50-43/tune_results
    GAMMA = 0.906        # İskonto faktörü | HPO: tune_ppo_phase1_2026-07-17_17-50-43/tune_results
    TRAIN_BATCH_SIZE = 1000
    ROLLOUT_FRAGMENT_LENGTH = 100
    
    # Model Architecture
    FCNET_HIDDENS = [512, 256]  # Huni mimarisi (daralan) | HPO: tune_ppo_phase1_2026-07-17_17-50-43/tune_results

class DQNConfig:
    """
    Parameters for Ray RLLib DQN Training.
    Used in train_dqn.py
    """
    # DQN Hyperparameters  |  Kaynak: tune_dqn_phase1_2026-07-18_18-03-23  |  Obj: 14.80
    LR    = 3.204e-4     # Öğrenme oranı   | HPO: tune_dqn_phase1_2026-07-18_18-03-23/optuna/best_params.json
    GAMMA = 0.872        # İskonto faktörü | HPO: tune_dqn_phase1_2026-07-18_18-03-23/optuna/best_params.json
    TRAIN_BATCH_SIZE = 1000
    ROLLOUT_FRAGMENT_LENGTH = 100
    
    # DQN-Specific Parameters
    TARGET_NETWORK_UPDATE_FREQ = 2000  # Her N adımda hedef ağ güncelleme | HPO: tune_dqn_phase1_2026-07-18_18-03-23/optuna/best_params.json
    DOUBLE_Q = True                    # Use Double DQN
    DUELING = True                     # Use Dueling DQN
    REPLAY_BUFFER_CAPACITY = 50000     # Replay buffer size
    
    # Training Control (CRITICAL for speed)
    NUM_STEPS_SAMPLED_BEFORE_LEARNING_STARTS = 0  # Start training after collecting initial samples
    TRAINING_INTENSITY = None           # Natural intensity (train_batch_size / (rollout_fragment_length * num_env_runners))
    
    # Model Architecture
    FCNET_HIDDENS = [128, 512, 512]  # Genişleyen mimari | HPO: tune_dqn_phase1_2026-07-18_18-03-23/optuna/best_params.json

class PPOLSTMConfig:
    """
    Parameters for PPO with LSTM (Recurrent Policy).
    Used in train_ppo_lstm.py
    """
    # PPO-LSTM Hyperparameters  |  Kaynak: tune_ppo_lstm_phase1_2026-07-18_19-28-57  |  Obj: 15.14
    LR    = 5.556e-4     # Öğrenme oranı   | HPO: tune_ppo_lstm_phase1_2026-07-18_19-28-57/optuna/best_params.json
    GAMMA = 0.865        # İskonto faktörü | HPO: tune_ppo_lstm_phase1_2026-07-18_19-28-57/optuna/best_params.json
    TRAIN_BATCH_SIZE = 1000
    ROLLOUT_FRAGMENT_LENGTH = 100
    
    # LSTM Specifics
    USE_LSTM = True
    LSTM_CELL_SIZE = 128  # LSTM hücre boyutu | HPO: tune_ppo_lstm_phase1_2026-07-18_19-28-57/optuna/best_params.json
    MAX_SEQ_LEN    = 20   # BPTT dizisi uzunluğu | HPO: tune_ppo_lstm_phase1_2026-07-18_19-28-57/optuna/best_params.json
    
    # Model Architecture
    FCNET_HIDDENS = [256, 512, 512]  # Genişleyen mimari | HPO: tune_ppo_lstm_phase1_2026-07-18_19-28-57/optuna/best_params.json


# Helper to dynamically override configurations with tuned parameters if they exist
def _load_tuned_configs():
    import os
    import json
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tuned_path = os.path.join(current_dir, "tuned_configs.json")
    if os.path.exists(tuned_path):
        try:
            with open(tuned_path, "r", encoding="utf-8") as f:
                tuned = json.load(f)
            
            # 1. PPO Overrides
            if "ppo" in tuned:
                p = tuned["ppo"]
                if "lr" in p: PPOConfig.LR = float(p["lr"])
                if "gamma" in p: PPOConfig.GAMMA = float(p["gamma"])
                if "architecture" in p:
                    arch_str = p["architecture"]
                    PPOConfig.FCNET_HIDDENS = [int(x) for x in arch_str.split(",") if x.strip()]
            
            # 2. DQN Overrides
            if "dqn" in tuned:
                d = tuned["dqn"]
                if "lr" in d: DQNConfig.LR = float(d["lr"])
                if "gamma" in d: DQNConfig.GAMMA = float(d["gamma"])
                if "target_network_update_freq" in d:
                    DQNConfig.TARGET_NETWORK_UPDATE_FREQ = int(d["target_network_update_freq"])
                if "architecture" in d:
                    arch_str = d["architecture"]
                    DQNConfig.FCNET_HIDDENS = [int(x) for x in arch_str.split(",") if x.strip()]
                    
            # 3. PPO-LSTM Overrides
            if "ppo_lstm" in tuned:
                pl = tuned["ppo_lstm"]
                if "lr" in pl: PPOLSTMConfig.LR = float(pl["lr"])
                if "gamma" in pl: PPOLSTMConfig.GAMMA = float(pl["gamma"])
                if "lstm_cell_size" in pl:
                    PPOLSTMConfig.LSTM_CELL_SIZE = int(pl["lstm_cell_size"])
                if "max_seq_len" in pl:
                    PPOLSTMConfig.MAX_SEQ_LEN = int(pl["max_seq_len"])
                if "architecture" in pl:
                    arch_str = pl["architecture"]
                    PPOLSTMConfig.FCNET_HIDDENS = [int(x) for x in arch_str.split(",") if x.strip()]
            
            # 4. QJC Overrides
            if "qjc" in tuned:
                q = tuned["qjc"]
                if "tau_0" in q: QJCConfig.TAU_0 = float(q["tau_0"])
                if "gamma" in q: QJCConfig.GAMMA = float(q["gamma"])
                if "temp_xi" in q: QJCConfig.TEMP_XI = float(q["temp_xi"])
                if "mu_offset" in q: QJCConfig.MU_OFFSET = float(q["mu_offset"])
                
            print(f"[OK] Loaded tuned hyperparameters dynamically from confs/tuned_configs.json")
        except Exception as e:
            print(f"[WARN] Failed to load tuned_configs.json: {e}")

# Run dynamic loader
_load_tuned_configs()
