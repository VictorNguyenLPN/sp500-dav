import os
import re
import math
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv # type: ignore
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor # type: ignore

np.random.seed(42)
pd.set_option("display.max_columns", None)

FIG_DIR = "fig"

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY_2")

raw_data = yf.download(
    tickers = "GSPC",
    start = "1900-01-01",
    end = "2025-01-01",
    interval = "1D",
    progress=False,
    auto_adjust=True
)

raw_data.columns = raw_data.columns.get_level_values(0)
raw_data = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']]
raw_data.reset_index(inplace=True)
raw_data.columns.name = None
raw_data['Date'] = pd.to_datetime(raw_data['Date'])

alias = {
    "VIXY": "VIX",
    "IEF": "US10Y",
    "UUP": "DXY",
    "USO": "OIL",
    "GLD": "GOLD",
    "CPI": "CPI_US"
}

def get_indicator(symbol, start, end):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, meta = ts.get_daily(symbol=symbol, outputsize='full')

    data = data.rename(columns={'4. close': 'Close'})
    data.index = pd.to_datetime(data.index)
    data = data.loc[(data.index >= start) & (data.index <= end)]

    data = data.reset_index()
    if 'index' in data.columns:
        data = data.rename(columns={'index': 'Date'})
    elif 'date' in data.columns:
        data = data.rename(columns={'date': 'Date'})
    else:
        data['Date'] = data.index

    clean_name = alias.get(symbol, re.sub(r'[^A-Za-z0-9_]+', '', symbol))
    data = data.rename(columns={'Close': clean_name})

    return data[['Date', clean_name]].dropna()

start_date = raw_data['Date'].min().strftime('%Y-%m-%d')
end_date   = raw_data['Date'].max().strftime('%Y-%m-%d')

symbols = ["VIXY", "IEF", "UUP", "USO", "GLD", "CPI"]

aug_data = raw_data.copy()
for sym in symbols:
    try:
        ind = get_indicator(sym, start_date, end_date)
        aug_data = pd.merge_asof(
            aug_data.sort_values('Date'),
            ind.sort_values('Date'),
            on='Date'
        )
        print(f"Merged {sym}")
    except Exception as e:
        print(f"Error when fetch {sym}: {e}")

delta = aug_data['Close'].diff()
abs_diff = delta.abs()

# MA, EMA, MAE
for w in [20, 60, 120, 360]:
    aug_data[f'MA_{w}'] = aug_data['Close'].rolling(window=w).mean()
    aug_data[f'EMA_{w}'] = aug_data['Close'].ewm(span=w, adjust=False).mean()
    aug_data[f'MAE_{w}'] = abs_diff.rolling(window=w).mean()

# MACD
ema12 = aug_data['Close'].ewm(span=12, adjust=False).mean()
ema26 = aug_data['Close'].ewm(span=26, adjust=False).mean()
ema60 = aug_data['Close'].ewm(span=60, adjust=False).mean()
ema120 = aug_data['Close'].ewm(span=120, adjust=False).mean()

aug_data['MACD'] = ema12 - ema26
aug_data['MACD_long'] = ema60 - ema120
aug_data['Signal'] = aug_data['MACD'].ewm(span=9, adjust=False).mean()
aug_data['Signal_long'] = aug_data['MACD_long'].ewm(span=45, adjust=False).mean()

# RSI
delta = aug_data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

for p in [7, 14, 28, 56]:
    avg_gain = gain.rolling(p).mean()
    avg_loss = loss.rolling(p).mean()
    rs = avg_gain / avg_loss
    aug_data[f'RSI_{p}'] = 100 - (100 / (1 + rs))
    
delta = aug_data['Close'].diff()
abs_diff = delta.abs()

# MA, EMA, MAE
for w in [20, 60, 120, 360]:
    aug_data[f'MA_{w}'] = aug_data['Close'].rolling(window=w).mean()
    aug_data[f'EMA_{w}'] = aug_data['Close'].ewm(span=w, adjust=False).mean()
    aug_data[f'MAE_{w}'] = abs_diff.rolling(window=w).mean()

# MACD
ema12 = aug_data['Close'].ewm(span=12, adjust=False).mean()
ema26 = aug_data['Close'].ewm(span=26, adjust=False).mean()
ema60 = aug_data['Close'].ewm(span=60, adjust=False).mean()
ema120 = aug_data['Close'].ewm(span=120, adjust=False).mean()

aug_data['MACD'] = ema12 - ema26
aug_data['MACD_long'] = ema60 - ema120
aug_data['Signal'] = aug_data['MACD'].ewm(span=9, adjust=False).mean()
aug_data['Signal_long'] = aug_data['MACD_long'].ewm(span=45, adjust=False).mean()

# RSI
delta = aug_data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

for p in [7, 14, 28, 56]:
    avg_gain = gain.rolling(p).mean()
    avg_loss = loss.rolling(p).mean()
    rs = avg_gain / avg_loss
    aug_data[f'RSI_{p}'] = 100 - (100 / (1 + rs))

first_valid_index = aug_data.dropna().index.min()
cropped_data = aug_data.loc[first_valid_index:].reset_index(drop=True)

cropped_data.to_csv("data/v6.csv")