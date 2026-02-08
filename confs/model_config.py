
class GlobalConfig:
    """
    Global configuration shared across all algorithms.
    Ensures reproducibility and consistency.
    """
    RANDOM_SEED = 42            # Random seed for reproducibility
    FLATTEN_ACTIONS = True      # Flatten MultiDiscrete to Discrete for DQN compatibility
    TRAIN_ITERATIONS = 20      # Number of training iterations
    TRAIN_BATCH_SIZE = 1000     # Steps collected per iteration (PPO/DQN)

class QJCConfig:
    """
    Parameters for the Baseline QJC Algorithm (Liao et al. 2025).
    Used in simulation/entities.py and train_baseline.py
    """
    TAU_0 = 0.1          # Base Learning Rate
    GAMMA = 0.9          # Discount Factor
    TEMP_XI = 5.0        # Softmax Temperature (Xi)
    MU_OFFSET = 1.1      # Offset for log calculation (mu + offset)
    TRAIN_EPISODES = GlobalConfig.TRAIN_ITERATIONS * 10  # Baseline episodes (200 episodes @ 100 steps each = 20k steps)
    SAVE_PATH = "baseline_q_table"          # Path to save Q-table
    MAX_POWER_LEVEL = 9                     # Maximum jammer power level (0-9)

class PPOConfig:
    """
    Parameters for Ray RLLib PPO Training.
    Used in train.py
    """

    # PPO Hyperparameters
    LR = 1e-4            # Learning Rate
    GAMMA = 0.9          # Discount Factor (harmonized with QJC/DQN)
    TRAIN_BATCH_SIZE = 1000
    ROLLOUT_FRAGMENT_LENGTH = 100
    
    # Model Architecture
    FCNET_HIDDENS = [256, 256]  # Fully connected hidden layers
    
    # Resources
    NUM_WORKERS = 1      # Number of parallel rollout workers
    USE_GPU = True       # GTX 3080 detected with CUDA 12.1

class DQNConfig:
    """
    Parameters for Ray RLLib DQN Training.
    Used in train_dqn.py
    """
    # DQN Hyperparameters
    LR = 1e-4              # Learning Rate
    GAMMA = 0.9            # Discount Factor (harmonized with QJC/PPO)
    TRAIN_BATCH_SIZE = 1000
    
    # DQN-Specific Parameters
    TARGET_NETWORK_UPDATE_FREQ = 500  # Update target network every N steps
    DOUBLE_Q = True                   # Use Double DQN
    DUELING = True                    # Use Dueling DQN
    REPLAY_BUFFER_CAPACITY = 50000    # Replay buffer size
    
    # Training Control (CRITICAL for speed)
    NUM_STEPS_SAMPLED_BEFORE_LEARNING_STARTS = 1000  # Start training after collecting initial samples
    TRAINING_INTENSITY = 1            # Gradient updates per env step (1 = match PPO speed)
    
    # Model Architecture
    FCNET_HIDDENS = [256, 256]  # Fully connected hidden layers
    
    # Resources
    NUM_WORKERS = 1      # Number of parallel rollout workers
    USE_GPU = True       # GTX 3080 detected with CUDA 12.1
