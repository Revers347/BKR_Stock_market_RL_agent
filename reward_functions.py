import numpy as np
from config import TRAINING_CONFIG


#reward based on returns 
def returns_reward(history, current_step):
    rets = history['Returns'].iloc[current_step]
    reward = rets
    if rets < 0:
        reward *= TRAINING_CONFIG['punishment_factor']
        if rets < -0.3:
            reward *= 10
    return np.clip(reward, -TRAINING_CONFIG['reward_clip'], TRAINING_CONFIG['reward_clip'])


#reward based on Sharpe ratio
def sharpe_reward(history, current_step, risk_free_rate=0.03, periods_per_year=252):
    period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    observed_returns = history['Returns'].iloc[:current_step + 1]
    excess_returns = observed_returns - period_rf

    mean_excess = np.mean(excess_returns)
    std_dev = np.std(observed_returns)

    sharpe_ratio = mean_excess / std_dev if std_dev > 0 else 0
    annual_sr = sharpe_ratio * np.sqrt(periods_per_year)

    return np.clip(annual_sr, -TRAINING_CONFIG['reward_clip'], TRAINING_CONFIG['reward_clip'])


#select type of reward
def get_reward_function(reward_type):

    if reward_type == 'sharpe':
        return sharpe_reward
    return returns_reward
