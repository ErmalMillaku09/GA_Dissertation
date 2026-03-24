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
    Sigma : np.array
        Annualized covariance matrix
    mean_returns : np.array
        Annualized mean returns
    prices : pd.DataFrame
        Historical prices for plotting
    """
    print(f"Downloading data for {tickers}...")

    # Download adjusted close prices
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Check if data is empty
    if data.empty:
        raise ValueError(f"No data downloaded for {tickers}. Check ticker symbols and date range.")

    # Use Adjusted Close if available, otherwise Close
    if 'Adj Close' in data:
        prices = data['Adj Close']
    elif 'Close' in data:
        prices = data['Close']
    else:
        raise ValueError("No price data available")

    # Calculate daily returns
    returns = prices.pct_change().dropna()

    if returns.empty:
        raise ValueError("Not enough price data to calculate returns")

    # Calculate annualized covariance (252 trading days)
    Sigma = returns.cov() * 252

    # Calculate annualized mean returns
    mean_returns = returns.mean() * 252

    print(f"Data downloaded: {returns.shape[0]} trading days")
    print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"Assets: {list(prices.columns)}")

    return Sigma.values, mean_returns.values, prices


def get_tech_stocks():
    """Return a list of tech stocks for portfolio optimization"""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']


def get_diverse_stocks():
    """Return a diverse set of stocks from different sectors"""
    return ['JPM', 'XOM', 'PG', 'JNJ', 'DIS', 'AAPL', 'BA', 'CAT']