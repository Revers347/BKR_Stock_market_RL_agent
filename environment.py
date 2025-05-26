from gym_trading_env.environments import TradingEnv
from reward_functions import get_reward_function
from config import ENV_CONFIG
 

def create_env(df, reward_type='returns'):
    def reward_fn(history):
        current_step = len(history['portfolio_valuation']) - 1
        return get_reward_function(reward_type)(history, current_step)
    
    env = TradingEnv(
        df=df,
        positions=ENV_CONFIG['positions'],
        trading_fees=ENV_CONFIG['trading_fees'],
        borrow_interest_rate=ENV_CONFIG['borrow_interest_rate'],
        reward_function=reward_fn,
        windows=ENV_CONFIG['window_size'],
        add_returns_to_features=True
    )
    return env
    