import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, List, Dict

pd.options.mode.chained_assignment = None  # Disable pandas warning noise

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Average True Range (ATR). This function is already efficient.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()


def detect_swing_points(df: pd.DataFrame, window: int = 3) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
    """
    Detects swing highs and lows using the highly efficient `scipy.signal.find_peaks`.
    This is significantly faster and more accurate than the original loop-based method.
    
    Args:
        df: DataFrame with OHLC data.
        window: The number of bars on each side of a swing point that must be lower/higher.
                Corresponds to the `distance` parameter in find_peaks.
    
    Returns:
        A tuple of lists containing (timestamp, price) for swing highs and swing lows.
    """
    atr = calculate_atr(df).mean()
    
    # Use ATR to set a dynamic prominence, filtering out insignificant noise.
    # A peak must stand out by at least 20% of the average ATR to be considered.
    required_prominence = atr * 0.20 if atr > 0 else (df['high'].max() - df['low'].min()) * 0.01

    # Find indices of peaks (highs) and troughs (lows)
    # The 'distance' parameter ensures swings are at least `window` bars apart.
    high_indices, _ = find_peaks(df['high'], prominence=required_prominence, distance=window)
    low_indices, _ = find_peaks(-df['low'], prominence=required_prominence, distance=window)

    # Format output to match the required List[Tuple[timestamp, price]] structure
    swing_highs = [(df.index[i], df['high'].iloc[i]) for i in high_indices]
    swing_lows = [(df.index[i], df['low'].iloc[i]) for i in low_indices]
    
    return swing_highs, swing_lows


def detect_trend(swing_highs: list[tuple[pd.Timestamp, float]], swing_lows: list[tuple[pd.Timestamp, float]]) -> str:
    """
    Determines trend by classifying the sequence of swing points into HH, HL, LH, LL.
    This provides a more robust trend definition.
    """
    if not swing_highs or not swing_lows:
        return "sideways"

    # Combine swings into a single DataFrame, add type, and sort by timestamp
    highs_df = pd.DataFrame(swing_highs, columns=['timestamp', 'price']).assign(type='high')
    lows_df = pd.DataFrame(swing_lows, columns=['timestamp', 'price']).assign(type='low')
    swings = pd.concat([highs_df, lows_df]).sort_values(by='timestamp').reset_index(drop=True)

    if len(swings) < 4:
        return "sideways" # Not enough structure to determine a trend

    # Classify the structure (HH, LH, HL, LL)
    classifications = []
    for i in range(1, len(swings)):
        current_swing = swings.iloc[i]
        
        # --- FIX: Replaced chained indexing with .loc to avoid UserWarning ---
        # Find the previous swing of the same type more efficiently
        mask = (swings.index < i) & (swings['type'] == current_swing['type'])
        prev_swings_of_type = swings.loc[mask]
        # --- End of fix ---
        
        if prev_swings_of_type.empty:
            continue
        
        prev_swing = prev_swings_of_type.iloc[-1]
        
        if current_swing['type'] == 'high':
            classification = 'HH' if current_swing['price'] > prev_swing['price'] else 'LH'
        else: # type is 'low'
            classification = 'HL' if current_swing['price'] > prev_swing['price'] else 'LL'
        
        classifications.append(classification)

    if not classifications:
        return "sideways"

    # Analyze the last 4 structure points for a clear trend
    recent_structure = classifications[-4:]
    
    is_uptrend = recent_structure.count('HH') + recent_structure.count('HL') >= 3
    is_downtrend = recent_structure.count('LL') + recent_structure.count('LH') >= 3

    if is_uptrend and not is_downtrend:
        return "uptrend"
    elif is_downtrend and not is_uptrend:
        return "downtrend"
    else:
        return "sideways"

def get_trend_from_data(resampled_data: dict[str, pd.DataFrame]) -> str:
    """
    Detects trends from resampled OHLC data across multiple timeframes.
    This function remains unchanged but now calls the improved methods.
    """
    trends = {}
    candles_to_use = {
        "4H": 360,   # ~60 days
        "1H": 720,   # ~30 days
        "15M": 960   # ~10 days
    }

    print("\n=== Trend Detection Detail ===")
    for tf in ["4H", "1H", "15M"]:
        df_full = resampled_data.get(tf)
        if df_full is None or df_full.empty:
            print(f"⚠️ No data for {tf}. Skipping.")
            trends[tf] = "sideways"
            continue

        df = df_full[-candles_to_use.get(tf, len(df_full)):]
        start_time, end_time = df.index[0], df.index[-1]

        print(f"\n🕒 {tf} timeframe from {start_time} to {end_time}")
        swing_highs, swing_lows = detect_swing_points(df)
        trend = detect_trend(swing_highs, swing_lows)
        trends[tf] = trend

        print(f"📊 {tf} Trend: {trend}")
        print(f"   ↪ Swing Highs: {len(swing_highs)}, Swing Lows: {len(swing_lows)}")

    # Override 4H trend if 1H and 15M agree and are not sideways
    final_trend = trends["4H"]
    if trends["1H"] != "sideways" and trends["1H"] == trends["15M"]:
        final_trend = trends["1H"]
        print(f"\n❗️ Override triggered: 1H and 15M trend ({trends['1H']}) is overriding 4H trend.")

    print("\n=== Final Trend Summary ===")
    for tf, trend in trends.items():
        print(f"{tf} Trend: {trend}")
    print(f"📌 Final Trend (with override logic): {final_trend}")

    return final_trend