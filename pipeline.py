import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def ingest_ohlc_data(data_source, db_file=None):
    """
    Ingest OHLC data from various sources.
    """
    if data_source == "CSV":
        df = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\Assignment\GOOGL_ohlc_data.csv")
    elif data_source == "JSON":
        df = yf.download("CL=F")
    elif data_source == "SQL":
        if db_file is None:
            raise ValueError("Database file path is required for SQL data source.")
        engine = sqlite3.connect(db_file)
        df = pd.read_sql("SELECT * FROM ohlc_data", engine)
    else:
        raise ValueError("Invalid data source. Please choose from CSV, JSON, or SQL.")

    data_check(df)
    logging.info("Data ingestion successful.")
    return df


def data_check(data):
    """
    Perform data validation checks.
    """
    assert not data.empty, "DataFrame is empty. Please check the data source."

    num_missing_values = data.isna().sum().sum()
    assert num_missing_values == 0, f"There are {num_missing_values} missing values in the data."

    assert data["Open"].dtype == np.float64, "Data type for Open needs to be float64."
    assert data["High"].dtype == np.float64, "Data type for High needs to be float64."
    assert data["Low"].dtype == np.float64, "Data type for Low needs to be float64."
    assert data["Close"].dtype == np.float64, "Data type for Close needs to be float64."

    logging.info("Data integrity check passed.")


def save_to_sqlite(data, db_file, table_name='ohlc_data'):
    """
    Save DataFrame to SQLite database.
    """
    engine = sqlite3.connect(db_file)
    data.to_sql(table_name, engine, if_exists='replace', index=False)


def calculate_sma(data, window=20):
    """
    Calculate Simple Moving Average (SMA) for a given window size.
    """
    return data['Close'].rolling(window=window).mean()


def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands (BB) for a given window size and number of standard deviations.
    """
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return upper_band, lower_band


def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI) for a given window size.
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_volatility_measures(data, window=20):
    """
    Calculate volatility measures such as standard deviation and ATR.
    """
    data['Std_Dev'] = data['Close'].rolling(window=window).std()
    high_low_range = data['High'] - data['Low']
    high_close_range = np.abs(data['High'] - data['Close'].shift())
    low_close_range = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=window).mean()
    return data


def calculate_price_patterns(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate price patterns such as MACD and EMA.
    """
    data['EMA_short'] = data['Close'].ewm(span=short_window, min_periods=1).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_window, min_periods=1).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['Signal_line'] = data['MACD'].ewm(span=signal_window, min_periods=1).mean()
    data['EMA_9'] = data['Close'].ewm(span=9, min_periods=1).mean()
    return data


def resample_ohlc_data(data, frequency='H'):
    """
    Resample OHLC data based on desired frequency.
    """
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    resampled_data = data.resample(frequency).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    return resampled_data


def partition_data(data):
    """
    Partition the OHLC data by year and month for efficient querying.
    """
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    grouped_data = data.groupby(['Year', 'Month'])
    return grouped_data


if __name__ == "__main__":
    # Example usage
    try:
        df = ingest_ohlc_data("CSV")
    except Exception as e:
        logging.error("Pipeline failed with error: %s", e)

    # Additional functionalities can be added here based on requirements.
