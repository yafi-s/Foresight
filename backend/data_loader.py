# ----------------------------------
# Stock Data Loading and Preprocessing
# ----------------------------------
# This module handles the loading, cleaning, and preprocessing of stock data.
# It includes functions for:
# - Fetching historical stock data
# - Feature engineering
# - Data normalization
# - Sequence creation for LSTM training

import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from typing import Optional, Tuple, List
import logging
from pathlib import Path
from sqlmodel import Session, select
from database import engine
from models import StockData
import numpy as np
import traceback
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from model import PREDICTION_DAYS  # Import PREDICTION_DAYS (should be 5)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for prediction
PREDICTION_DAYS = 5  # Number of days to predict into the future

def get_latest_stock_data(symbol: str) -> Optional[datetime]:
    """Return the most recent date we have for <symbol> in the database."""
    with Session(engine) as session:
        statement = (
            select(StockData)
            .where(StockData.symbol == symbol)
            .order_by(StockData.date.desc())
        )
        result = session.exec(statement).first()
        return result.date if result else None

def save_stock_data(symbol: str, df: pd.DataFrame) -> None:
    """Persist rows of `df` into the StockData table."""
    with Session(engine) as session:
        # Filter out rows with NaN values in any of the required columns
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        for index, row in df.iterrows():
            stock_data = StockData(
                symbol=symbol,
                date=index.to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
            session.add(stock_data)
        session.commit()

def load_stock_data(
    symbol: str,
    period: str = "5y",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Load historical stock data for a given symbol.
    Args:
        symbol: Stock ticker symbol
        period: Time period to fetch (default: 5 years)
        force_refresh: Whether to force data refresh
    Returns:
        DataFrame containing historical stock data
    """
    latest_date = get_latest_stock_data(symbol)
    now = datetime.now()

    # If DB has data within the last 24 h, just load from DB
    if not force_refresh and latest_date and (now - latest_date) < timedelta(hours=24):
        logger.info(f"Loading {symbol} data from database (cached).")
        with Session(engine) as session:
            statement = (
                select(StockData)
                .where(StockData.symbol == symbol)
                .order_by(StockData.date)
            )
            results = session.exec(statement).all()
            if results:
                df = pd.DataFrame(
                    [
                        {
                            "Open": r.open,
                            "High": r.high,
                            "Low": r.low,
                            "Close": r.close,
                            "Volume": r.volume,
                        }
                        for r in results
                    ],
                    index=[r.date for r in results],
                )
                df.index = pd.to_datetime(df.index)
                # Handle duplicate dates by keeping the most recent entry
                df = df[~df.index.duplicated(keep='last')]
                return df

    # Otherwise, fetch from YahooFinance
    logger.info(f"Downloading {symbol} data from yfinance.")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d")
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol} on yfinance.")
        
        # Handle duplicate dates by keeping the most recent entry
        df = df[~df.index.duplicated(keep='last')]
        
        # Save new data to DB
        save_stock_data(symbol, df)
        return df
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {e}")
        if latest_date:
            # Fallback to whatever is in the DB
            logger.info("Falling back to stored database data.")
            with Session(engine) as session:
                statement = (
                    select(StockData)
                    .where(StockData.symbol == symbol)
                    .order_by(StockData.date)
                )
                results = session.exec(statement).all()
                if results:
                    df = pd.DataFrame(
                        [
                            {
                                "Open": r.open,
                                "High": r.high,
                                "Low": r.low,
                                "Close": r.close,
                                "Volume": r.volume,
                            }
                            for r in results
                        ],
                        index=[r.date for r in results],
                    )
                    df.index = pd.to_datetime(df.index)
                    # Handle duplicate dates by keeping the most recent entry
                    df = df[~df.index.duplicated(keep='last')]
                    return df
        # If no fallback, re‐raise
        raise

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.ewm(com=window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(
    df: pd.DataFrame,
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9
) -> Tuple[pd.Series, pd.Series]:
    """Calculate the Moving Average Convergence Divergence (MACD)."""
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate the Average True Range (ATR)."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.ewm(span=window, adjust=False).mean()
    return atr

def prepare_features(
    df: pd.DataFrame, *, include_targets: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepare features and target variables for model training.
    Creates technical indicators and normalizes the data.
    Args:
        df: Raw stock data DataFrame
    Returns:
        Tuple of (feature DataFrame, target DataFrame, feature column names)
    """
    original_closes = df["Close"].copy()
    # Calculate returns
    df['Returns'] = df['Close'].pct_change(fill_method=None)
    
    # Calculate volatility metrics
    df['Volatility'] = df['Returns'].rolling(window=5).std()  # 5-day volatility
    
    # Handle potential division by zero in High_Low_Range
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close'].replace(0, np.nan)
    
    # Handle potential division by zero in Volume_Change
    df['Volume_Change'] = df['Volume'].pct_change(fill_method=None)
    
    # Calculate price momentum indicators
    df['Price_Momentum'] = df['Close'].pct_change(periods=5, fill_method=None)
    df['Volume_Momentum'] = df['Volume'].pct_change(periods=5, fill_method=None)
    
    # Calculate Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
    
    # Handle potential division by zero in BB_Position
    bb_range = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / bb_range.replace(0, np.nan)
    
    # Add MA5 (5-day moving average)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    
    # Replace any remaining infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop NaN values
    df = df.dropna()
    
    # Select features - exactly as used during training, in the correct order
    feature_cols = [
        'Returns', 'Volatility', 'High_Low_Range', 'Volume_Change',
        'Price_Momentum', 'Volume_Momentum', 'BB_Position', 'MA5'
    ]
    X_df = df[feature_cols]
    
    # Debug logging for feature columns and shape
    logger.info(f"prepare_features: Feature columns: {list(X_df.columns)}")
    logger.info(f"prepare_features: Feature shape: {X_df.shape}")

    if not include_targets:
        # Serving has no future labels and must retain the latest observations.
        return X_df, pd.DataFrame(index=X_df.index), feature_cols
    
    # Prepare y (target) for multiple days ahead
    y_df = pd.DataFrame(index=df.index)
    for d in range(1, PREDICTION_DAYS + 1):
        y_df[f'Future_Return_{d}'] = (original_closes.shift(-d) / original_closes - 1).reindex(df.index)
    
    # Retain feature history independently of future-label availability.
    # causal.prepare_split excludes only origins with incomplete target vectors.
    
    return X_df, y_df, feature_cols

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    Args:
        X: Feature array
        y: Target array
        sequence_length: Length of input sequences
        step: Step size for sequence creation
    Returns:
        Tuple of (input sequences, target sequences)
    """
    X_seq, y_seq = [], []
    
    for i in range(0, len(X) - sequence_length, step):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])
    
    return np.array(X_seq), np.array(y_seq)
