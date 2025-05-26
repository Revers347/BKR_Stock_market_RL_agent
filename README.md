Trading RL Agent

This project implements a Reinforcement Learning (RL) agent for stock trading using the DQN algorithm. The agent is trained on historical stock data with technical and fundamental indicators, then evaluated on test data.
Features

    Data preprocessing with technical (MACD, ATR) and financial indicators

    Custom trading environment with trading fees and borrowing interest

    Multiple reward functions (returns and Sharpe ratio)

    Model training with Stable Baselines3 DQN

    Visualization of trading actions and portfolio performance

    Performance metrics: cumulative return, annual return, Sharpe and Sortino ratios


Configuration
Parameters for data, environment, model, and training are set in config.py.

Run training and evaluation:
python training.py