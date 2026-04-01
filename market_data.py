# market_data.py
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def download_stock_data(tickers, start_date, end_date):
    """
    Download stock data and compute annualized covariance matrix.

    Parameters:
    -----------
    tickers : list
        List of stock symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    start_date : str
        'YYYY-MM-DD'
    end_date : str
        'YYYY-MM-DD'

    Returns:
    --------
    tickers : list
        List of successfully downloaded tickers
    Sigma : np.array
        Annualized covariance matrix
    mean_returns : np.array
        Annualized mean returns
    prices : pd.DataFrame
        Historical prices for plotting
    """
    print(f"Downloading data for {len(tickers)} stocks...")

    # Download adjusted close prices with error handling
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"Warning: Download had issues: {e}")
        data = None

    # Check if data is empty
    if data is None or data.empty:
        raise ValueError(f"No data downloaded for {tickers}. Check ticker symbols and date range.")

    # Use Adjusted Close if available, otherwise Close
    if 'Adj Close' in data:
        prices = data['Adj Close']
    elif 'Close' in data:
        prices = data['Close']
    else:
        raise ValueError("No price data available")

    # Drop columns with all NaN or mostly NaN (failed downloads or bad data)
    # Keep columns that have at least 80% of data
    min_non_null = len(prices) * 0.8
    prices = prices.dropna(axis=1, thresh=min_non_null)

    if prices.empty:
        raise ValueError("No valid price data available after filtering")

    # Forward fill then backward fill to handle small gaps (newer pandas syntax)
    prices = prices.ffill().bfill()

    # Calculate daily returns
    returns = prices.pct_change().dropna()

    if returns.empty or len(returns) < 100:
        raise ValueError("Not enough valid price data to calculate returns")

    # Calculate annualized covariance (252 trading days)
    Sigma = returns.cov() * 252

    # Calculate annualized mean returns
    mean_returns = returns.mean() * 252

    valid_tickers = list(prices.columns)
    print(f"\nData downloaded successfully!")
    print(f"  Valid tickers: {len(valid_tickers)}/{len(tickers)}")
    print(f"  Trading days: {returns.shape[0]}")
    print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Assets: {valid_tickers}")

    return valid_tickers, Sigma.values, mean_returns.values, prices


def get_tech_stocks():
    """Return a list of tech stocks for portfolio optimization"""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']


def get_diverse_stocks():
    """Return a diverse set of stocks from different sectors"""
    return ['JPM', 'XOM', 'PG', 'JNJ', 'DIS', 'AAPL', 'BA', 'CAT']


def get_52_stocks():
    """Return a list of 50 reliable stocks for portfolio optimization"""
    return [
        # Large cap tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
        # Financial
        'JPM', 'BAC', 'GS', 'MS', 'WFC', 'BLK', 'SPGI',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT',
        # Industrials/Consumer
        'HD', 'WMT', 'MCD', 'COST', 'DIS', 'BA', 'CAT', 'HON',
        # Energy/Utilities
        'XOM', 'CVX', 'NEE', 'DUK', 'SO',
        # Semiconductors
        'AMD', 'QCOM', 'INTC', 'MU', 'AVGO',
        # Communications
        'VZ', 'T', 'CMCSA',
        # Consumer/Discretionary
        'NFLX', 'ADBE', 'CRM', 'PYPL', 'V', 'MA',
        # Industrial
        'UNP', 'IQV', 'GILD'
    ]
