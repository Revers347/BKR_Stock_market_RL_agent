import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from config import DATA_CONFIG, TRAINING_CONFIG


def download_data(ticker, start_date, end_date, filename):
    if os.path.exists(filename):
        print(f"Load data from file {filename}")
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        print(f"Download from Yahoo Finance {ticker}")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")
        data.to_csv(filename)
    return data

def add_technical_indicators(df):
    if TRAINING_CONFIG['use_technical']:
        for indicator in TRAINING_CONFIG['tech_indicators']:
            if indicator == 'MACD':
                macd = ta.macd(df['Close'])
                df = pd.concat([df, macd], axis=1)
            elif indicator == 'ATR':
                atr = ta.atr(df['High'], df['Low'], df['Close'])
                df['ATR'] = atr
    return df.dropna()

def add_fundamental_data(df, benchmarks):
    if TRAINING_CONFIG['use_fundamental']:
        for name, ticker in benchmarks.items():
            benchmark_data = yf.download(ticker, start=df.index[0], end=df.index[-1], progress=False)['Close']
            df[name] = benchmark_data.values
    return df.dropna()


def preprocess_data():
    train_filename = f"train_data_{DATA_CONFIG['ticker']}_{DATA_CONFIG['train_start'].strftime('%Y-%m-%d')}_{DATA_CONFIG['train_end'].strftime('%Y-%m-%d')}.csv"
    test_filename = f"test_data_{DATA_CONFIG['ticker']}_{DATA_CONFIG['test_start'].strftime('%Y-%m-%d')}_{DATA_CONFIG['test_end'].strftime('%Y-%m-%d')}.csv"
    train_data = download_data(DATA_CONFIG['ticker'], DATA_CONFIG['train_start'], DATA_CONFIG['train_end'], train_filename)
    test_data = download_data(DATA_CONFIG['ticker'], DATA_CONFIG['test_start'], DATA_CONFIG['test_end'], test_filename)
    
    if 'MACD_12' not in train_data.columns or 'ATR' not in train_data.columns:
        train_data = add_technical_indicators(train_data)
        test_data = add_technical_indicators(test_data)

    if len(DATA_CONFIG['benchmarks']) > 0 and not all(benchmark in train_data.columns for benchmark in DATA_CONFIG['benchmarks'].keys()):
        train_data = add_fundamental_data(train_data, DATA_CONFIG['benchmarks'])
        test_data = add_fundamental_data(test_data, DATA_CONFIG['benchmarks'])

    train_data['Returns'] = train_data['Close'].pct_change()
    test_data['Returns'] = test_data['Close'].pct_change()


    train_data.to_csv(train_filename)
    test_data.to_csv(test_filename)
    
    return train_data.dropna(), test_data.dropna()
