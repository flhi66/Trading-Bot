import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # Disable pandas warning noise
from typing import Tuple, List
from datetime import timedelta


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Average True Range (ATR) for volatility-based distance.
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

    atr = tr.rolling(window=period).mean()
    return atr


def detect_swing_points(df: pd.DataFrame, window: int = 3) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
    """
    Detect swing highs and lows using a windowed peak/trough approach with ATR-based spacing filter.
    """
    swing_highs = []
    swing_lows = []

    highs = df['high']
    lows = df['low']
    atr = calculate_atr(df).dropna()

    if atr.empty:
        min_distance = (highs.max() - lows.min()) * 0.2  # fallback
    else:
        min_distance = atr.mean() * 0.5

    for i in range(window, len(df) - window):
        is_swing_high = all(highs[i] > highs[i - j] and highs[i] > highs[i + j] for j in range(1, window + 1))
        is_swing_low = all(lows[i] < lows[i - j] and lows[i] < lows[i + j] for j in range(1, window + 1))

        if is_swing_high:
            if not swing_highs or abs(highs[i] - swing_highs[-1][1]) > min_distance:
                swing_highs.append((df.index[i], highs[i]))

        if is_swing_low:
            if not swing_lows or abs(lows[i] - swing_lows[-1][1]) > min_distance:
                swing_lows.append((df.index[i], lows[i]))

    return swing_highs, swing_lows


def detect_trend(swing_highs: list[tuple[pd.Timestamp, float]], swing_lows: list[tuple[pd.Timestamp, float]]) -> str:
    """
    Determines trend based on swing structure: 2 HH & 2 HL = uptrend, 2 LL & 2 LH = downtrend.
    """
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return "sideways"

    last_highs = swing_highs[-3:]
    last_lows = swing_lows[-3:]

    is_uptrend = (
        last_highs[2][1] > last_highs[1][1] > last_highs[0][1] and
        last_lows[2][1] > last_lows[1][1] > last_lows[0][1]
    )

    is_downtrend = (
        last_highs[2][1] < last_highs[1][1] < last_highs[0][1] and
        last_lows[2][1] < last_lows[1][1] < last_lows[0][1]
    )

    if is_uptrend:
        return "uptrend"
    elif is_downtrend:
        return "downtrend"
    else:
        return "sideways"


def get_trend_from_data(resampled_data: dict[str, pd.DataFrame]) -> str:
    """
    Detects trends from resampled OHLC data across multiple timeframes.
    Returns the final trend after applying override logic.
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

        # Trim to recent candles
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

    print("\n=== Final Trend Summary ===")
    for tf, trend in trends.items():
        print(f"{tf} Trend: {trend}")
    print(f"📌 Final Trend (with override logic): {final_trend}")

    return final_trend
