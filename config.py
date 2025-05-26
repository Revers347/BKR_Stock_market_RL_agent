from datetime import datetime

# Data configuration
DATA_CONFIG = {
    'ticker': 'F',          # Ford
    'benchmarks': {
        'SPX': '^GSPC',     # S&P 500 index
        'TNX': '^TNX',      # 10-year treasury yield
        'VIX': '^VIX'       # Volatility index
    },
    'train_start': datetime(2016, 1, 1),
    'train_end': datetime(2022, 4, 1),
    'test_start': datetime(2022, 4, 1),
    'test_end': datetime(2023, 12, 31),
    'split_date': datetime(2022, 4, 1)
}

# Environment configuration
ENV_CONFIG = {
    'positions': [0, 1, -1],       # 0: hold, 1: long, -1: short
    'trading_fees': 0.0001,        # Proportional trading fee
    'borrow_interest_rate': 0.00005,
    'window_size': 30              # Number of days in observations
}


# Model configuration (hyperparameters for DQN)
MODEL_CONFIG = {
    'policy': 'MlpPolicy',
    'learning_rate': 0.0003,
    'buffer_size': 100000,
    'learning_starts': 10000,
    'batch_size': 128,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': (4, 'step'),
    'gradient_steps': 1,
    'target_update_interval': 1000
}

# Training configuration
TRAINING_CONFIG = {
    'total_timesteps': 10000,
    'reward_types': ['returns', 'sharpe'],  # 2 different reward func.
    'use_technical': True,                  # Include technical indicators
    'use_fundamental': True,                # Include finansial data
    'tech_indicators': ['MACD', 'ATR'],
    'punishment_factor': 1.4,
    'reward_clip': 3.0
}
