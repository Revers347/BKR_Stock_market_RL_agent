from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from data_loader import preprocess_data
from environment import create_env
from visualization import plot_trading_results, calculate_metrics, print_metrics
from config import *


def evaluate_model(model, env, ticker):
    history = {
        'date': [],
        'price': [],
        'position': [],
        'portfolio_valuation': []
    }
    
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        history['date'].append(info['date'])
        history['price'].append(info['price'])
        history['position'].append(info['position'])
        history['portfolio_valuation'].append(info['portfolio_valuation'])
    
    plot_trading_results(history, ticker)
    metrics = calculate_metrics(history)
    print_metrics(metrics)
    
    return metrics


def run_experiment():
    train_df, test_df = preprocess_data()

    for reward_type in TRAINING_CONFIG['reward_types']:
        print(f"\nTraining with {reward_type} reward")

        train_env = DummyVecEnv([lambda: create_env(train_df, reward_type)])
        test_env = DummyVecEnv([lambda: create_env(test_df, reward_type)])

        model = DQN(
            MODEL_CONFIG['policy'],
            train_env,
            learning_rate=MODEL_CONFIG['learning_rate'],
            buffer_size=MODEL_CONFIG['buffer_size'],
            learning_starts=MODEL_CONFIG['learning_starts'],
            batch_size=MODEL_CONFIG['batch_size'],
            tau=MODEL_CONFIG['tau'],
            gamma=MODEL_CONFIG['gamma'],
            train_freq=MODEL_CONFIG['train_freq'],
            gradient_steps=MODEL_CONFIG['gradient_steps'],
            target_update_interval=MODEL_CONFIG['target_update_interval'],
            verbose=1
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path='./models/',
            name_prefix='dqn_model'
        )

        model.learn(total_timesteps=TRAINING_CONFIG['total_timesteps'], callback=checkpoint_callback)
        model.save("./models/dqn_model_final")

        evaluate_model(model, test_env, DATA_CONFIG['ticker'])


if __name__ == "__main__":
    run_experiment()
