import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter


def plot_trading_results(history, ticker):
    dates = history['date']
    prices = history['price']
    actions = ['BUY' if x == 1 else 'SELL' if x == -1 else 'HOLD' for x in history['position']]
    portfolio = history['portfolio_valuation']
    plt.figure(figsize=(14, 6))
    plt.plot(dates, prices, label=f'Ціна {ticker}', alpha=0.7)
    buy_dates = [dates[i] for i, a in enumerate(actions) if a == 'BUY']
    buy_prices = [prices[i] for i, a in enumerate(actions) if a == 'BUY']
    sell_dates = [dates[i] for i, a in enumerate(actions) if a == 'SELL']
    sell_prices = [prices[i] for i, a in enumerate(actions) if a == 'SELL']
    
    plt.scatter(buy_dates, buy_prices, color='green', label='Buy', marker='^', alpha=0.8)
    plt.scatter(sell_dates, sell_prices, color='red', label='Sell', marker='v', alpha=0.8)
    plt.title(f'Дії агента та ціна {ticker}')
    plt.xlabel('Дата')
    plt.ylabel('Ціна ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(14, 5))
    plt.plot(dates, portfolio, label='Кумулятивний прибуток', color='blue')
    plt.title('Кумулятивний прибуток агента')
    plt.xlabel('Дата')
    plt.ylabel('Портфель ($)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(portfolio[0], color='gray', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.tight_layout()
    plt.show()


def calculate_metrics(history, risk_free_rate=0.03):
    returns = pd.Series(history['portfolio_valuation']).pct_change().dropna()
    n_days = len(returns)
    total_return = (history['portfolio_valuation'][-1] / history['portfolio_valuation'][0] - 1) * 100
    annual_return = (1 + total_return/100)**(252/n_days) - 1 if n_days > 0 else 0
    excess_returns = returns - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    downside_returns = returns[returns < 0]
    sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 0 else 0
    
    actions = [1 if x == 1 else -1 if x == -1 else 0 for x in history['position']]
    buy_ops = actions.count(1)
    sell_ops = actions.count(-1)
    

    if buy_ops != sell_ops:
        print(f"  (Buy: {buy_ops}, Sell: {sell_ops})")
    
    return {
        "Cumulative Return (%)": round(total_return, 2),
        "Annual Return (%)": round(annual_return * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Sortino Ratio": round(sortino_ratio, 2),
        "Number of Buys": buy_ops,
        "Number of Sells": sell_ops,
        "Total Trades": buy_ops + sell_ops
    }


def print_metrics(metrics):
    print("\n Metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")