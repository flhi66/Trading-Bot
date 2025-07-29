import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional

# Assuming calculate_atr is defined elsewhere as in the original code.
# For demonstration, a placeholder is included.
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR) as a placeholder."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def detect_swing_points_scipy(df: pd.DataFrame, prominence_factor: float = 2.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detects swing high and low points using scipy.signal.find_peaks for better accuracy.

    Args:
        df: DataFrame with 'high' and 'low' columns.
        prominence_factor: Multiplier for the mean ATR to determine peak prominence.
                         A higher value means only more significant swings are detected.

    Returns:
        A tuple containing two DataFrames: swing_highs and swing_lows.
    """
    atr = calculate_atr(df)
    # Use a dynamic prominence based on ATR
    required_prominence = atr.mean() * prominence_factor
    
    # Invert low prices to find peaks, which are actually troughs
    lows_inv = -df['low']

    # Find peaks (swing highs) and troughs (swing lows)
    high_indices, high_props = find_peaks(df['high'], prominence=required_prominence)
    low_indices, low_props = find_peaks(lows_inv, prominence=required_prominence)

    swing_highs = df.iloc[high_indices][['high']].copy()
    swing_highs.rename(columns={'high': 'price'}, inplace=True)
    swing_highs['type'] = 'high'

    swing_lows = df.iloc[low_indices][['low']].copy()
    swing_lows.rename(columns={'low': 'price'}, inplace=True)
    swing_lows['type'] = 'low'
    
    return swing_highs, swing_lows

def build_market_structure(df: pd.DataFrame, prominence_factor: float = 2.0) -> List[Dict]:
    """
    Builds and classifies the market structure (HH, LH, HL, LL) chronologically.

    Args:
        df: The input price data.
        prominence_factor: Factor for swing detection prominence.

    Returns:
        A list of dictionaries, where each dictionary represents a classified structure point.
    """
    swing_highs, swing_lows = detect_swing_points_scipy(df, prominence_factor)
    
    # Combine swings and sort them by date to process in order
    all_swings = pd.concat([swing_highs, swing_lows]).sort_index()

    if len(all_swings) < 2:
        return []

    structure = []
    last_high: Optional[Dict] = None
    last_low: Optional[Dict] = None

    for timestamp, row in all_swings.iterrows():
        price = row['price']
        point_type = row['type']
        
        classification = None
        if point_type == 'high':
            if last_high:
                classification = "HH" if price > last_high['price'] else "LH"
            last_high = {'timestamp': timestamp, 'price': price}
        else: # point_type == 'low'
            if last_low:
                classification = "HL" if price > last_low['price'] else "LL"
            last_low = {'timestamp': timestamp, 'price': price}
        
        if classification:
            structure.append({
                "timestamp": timestamp,
                "type": classification,
                "price": price
            })
            
    return structure

def get_market_analysis(df: pd.DataFrame, prominence_factor: float = 2.0, trend_window: int = 4) -> Dict:
    """
    Main function to get market structure and confirm the current trend.

    Args:
        df: The input price data.
        prominence_factor: Controls the sensitivity of swing detection.
        trend_window: How many of the most recent structure points to consider for trend confirmation.

    Returns:
        A dictionary containing the full structure, swing points, and the determined trend.
    """
    structure = build_market_structure(df, prominence_factor)
    
    trend = "sideways"
    if len(structure) >= trend_window:
        recent_structure = structure[-trend_window:]
        types = [p['type'] for p in recent_structure]
        
        # More robust trend confirmation: look for sequences
        is_uptrend = "HH" in types and "HL" in types and "LL" not in types
        is_downtrend = "LL" in types and "LH" in types and "HH" not in types

        if is_uptrend:
            trend = "uptrend"
        elif is_downtrend:
            trend = "downtrend"

    # For returning the raw swing points as in the original function
    swing_highs, swing_lows = detect_swing_points_scipy(df, prominence_factor)
    
    return {
        "trend": trend,
        "structure": structure,
        "swing_highs": list(swing_highs.reset_index().to_records(index=False)),
        "swing_lows": list(swing_lows.reset_index().to_records(index=False))
    }

# --- Example Usage ---
# Create a sample DataFrame
# data = {
#     'high': [10, 12, 11, 13, 12, 15, 14, 13, 16, 15, 12, 11, 13, 10],
#     'low':  [8,  9,  8,  10, 11, 12, 11, 12, 13, 14, 10,  9,  11, 8],
#     'close':[9, 11, 9,  12, 11, 14, 13, 12, 15, 14, 11, 10, 12, 9]
# }
# index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(data['high'])))
# df = pd.DataFrame(data, index=index)
#
# analysis = get_market_analysis(df, prominence_factor=1.0)
# import json
# print(json.dumps(analysis, indent=2, default=str))