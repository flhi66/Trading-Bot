import pandas as pd
from typing import List, Dict, Tuple
from core.trend_detector import calculate_atr


def detect_swing_points(df: pd.DataFrame, window: int = 2) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
    swing_highs = []
    swing_lows = []

    highs = df['high']
    lows = df['low']
    atr = calculate_atr(df).dropna()

    if atr.empty:
        min_distance = (highs.max() - lows.min()) * 0.2
    else:
        min_distance = atr.mean() * 0.5

    for i in range(window, len(df) - window):
        is_high = all(highs.iloc[i] > highs.iloc[i - j] and highs.iloc[i] > highs.iloc[i + j] for j in range(1, window + 1))
        is_low = all(lows.iloc[i] < lows.iloc[i - j] and lows.iloc[i] < lows.iloc[i + j] for j in range(1, window + 1))

        if is_high:
            if not swing_highs or abs(highs.iloc[i] - swing_highs[-1][1]) > min_distance:
                swing_highs.append((df.index[i], highs.iloc[i]))

        if is_low:
            if not swing_lows or abs(lows.iloc[i] - swing_lows[-1][1]) > min_distance:
                swing_lows.append((df.index[i], lows.iloc[i]))

    return swing_highs, swing_lows


def build_structure(swing_highs: List[Tuple[pd.Timestamp, float]], swing_lows: List[Tuple[pd.Timestamp, float]]) -> List[Dict]:
    structure = []
    highs = iter(sorted(swing_highs))
    lows = iter(sorted(swing_lows))

    last_high = next(highs, None)
    last_low = next(lows, None)

    if not last_high or not last_low:
        return structure

    turn = 'high' if last_high[0] < last_low[0] else 'low'
    prev_high = last_high
    prev_low = last_low

    while True:
        if turn == 'high':
            current = next(highs, None)
            if current is None:
                break
            curr_type = "HH" if current[1] > prev_high[1] else "LH"
            structure.append({"timestamp": current[0], "type": curr_type, "price": current[1]})
            prev_high = current
            turn = 'low'

        else:
            current = next(lows, None)
            if current is None:
                break
            curr_type = "HL" if current[1] > prev_low[1] else "LL"
            structure.append({"timestamp": current[0], "type": curr_type, "price": current[1]})
            prev_low = current
            turn = 'high'

    return structure


def confirm_trend(structure: List[Dict], window: int = 8) -> str:
    if len(structure) < window:
        return "sideways"
    recent = structure[-window:]
    types = [point["type"] for point in recent]
    up_count = types.count("HH") + types.count("HL")
    down_count = types.count("LL") + types.count("LH")

    if up_count > down_count:
        return "uptrend"
    elif down_count > up_count:
        return "downtrend"
    return "sideways"


def get_structure(df: pd.DataFrame) -> Tuple[List[Dict], Dict]:
    swing_highs, swing_lows = detect_swing_points(df)
    structure = build_structure(swing_highs, swing_lows)
    trend = confirm_trend(structure)
    return structure, {"trend": trend, "swing_highs": swing_highs, "swing_lows": swing_lows}
