import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def find_pivots(df: pd.DataFrame, prominence: float = 0.01) -> pd.DataFrame:
    """
    Detects swing high and low pivot points in the price data.

    Args:
        df (pd.DataFrame): DataFrame with 'High' and 'Low' columns.
        prominence (float): The required prominence of the peaks. A higher value
                         means fewer, more significant pivots will be detected.

    Returns:
        pd.DataFrame: A DataFrame containing the timestamps and prices of all pivots, sorted by time.
    """
    # Use a percentage of the total price range to determine prominence
    price_range = df['High'].max() - df['Low'].min()
    required_prominence = price_range * prominence

    # Find peak and trough indices
    high_peaks, _ = find_peaks(df['High'], prominence=required_prominence)
    low_troughs, _ = find_peaks(-df['Low'], prominence=required_prominence)

    # Create DataFrames for highs and lows
    pivot_highs = pd.DataFrame({'price': df['High'].iloc[high_peaks]}, index=df.index[high_peaks])
    pivot_lows = pd.DataFrame({'price': df['Low'].iloc[low_troughs]}, index=df.index[low_troughs])
    
    # Combine, sort, and remove any duplicates that might occur at the same timestamp
    pivots = pd.concat([pivot_highs, pivot_lows]).sort_index()
    pivots = pivots[~pivots.index.duplicated(keep='first')]

    # Add the first and last data points to ensure the zig-zag line covers the whole chart
    first_point = pd.DataFrame({'price': df['Close'].iloc[[0]]}, index=df.index[[0]])
    last_point = pd.DataFrame({'price': df['Close'].iloc[[-1]]}, index=df.index[[-1]])
    
    pivots = pd.concat([first_point, pivots, last_point])
    pivots = pivots[~pivots.index.duplicated(keep='first')].sort_index()
    
    return pivots