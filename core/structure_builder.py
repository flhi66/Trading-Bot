import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional

# Assumes this function is also updated or exists elsewhere.
def calculate_atr(df: pd.DataFrame, period: int = 14, exponential_mean: bool = True) -> pd.Series:
    """Calculate the Average True Range (ATR) with optional exponential smoothing."""
    high, low, close = df['High'], df['Low'], df['Close']

    tr = np.maximum.reduce([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ])
    tr = pd.Series(tr, index=df.index)

    if not exponential_mean:
        atr = tr.rolling(window=period).mean()
    else:
        atr = tr.ewm(alpha=1/period, adjust=False).mean()

    return atr

def detect_swing_points_scipy(df: pd.DataFrame, prominence_factor: float = 7.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detects swing points using scipy, now with capitalized column names.
    """
    atr = calculate_atr(df)
    required_prominence = atr.mean() * prominence_factor if pd.notna(atr.mean()) and atr.mean() > 0 else (df['High'].max() - df['Low'].min()) * 0.01
    
    # FIX: Changed column names to capitalized versions
    lows_inv = -df['Low']
    high_indices, _ = find_peaks(df['High'], prominence=required_prominence)
    low_indices, _ = find_peaks(lows_inv, prominence=required_prominence)

    swing_highs = df.iloc[high_indices][['High']].copy()
    swing_highs.rename(columns={'High': 'price'}, inplace=True) # FIX: Renaming from 'High'
    swing_highs['type'] = 'high'

    swing_lows = df.iloc[low_indices][['Low']].copy()
    swing_lows.rename(columns={'Low': 'price'}, inplace=True) # FIX: Renaming from 'Low'
    swing_lows['type'] = 'low'
    
    return swing_highs, swing_lows

def detect_swing_points_by_atr(df: pd.DataFrame, prominence_factor: float = 7.5): #-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detects swing points using scipy, now with capitalized column names.
    """
    atr = calculate_atr(df)
    required_prominence = atr.mean() * prominence_factor if pd.notna(atr.mean()) and atr.mean() > 0 else (df['High'].max() - df['Low'].min()) * 0.01
    
    new_df = pd.DataFrame(index=df.index, columns=["HighLow", "Level"])

    lows_inv = -df['Low']
    high_indices, _ = find_peaks(df['High'], prominence=required_prominence)
    low_indices, _ = find_peaks(lows_inv, prominence=required_prominence)

    swing_highs = df.iloc[high_indices][['High']].copy()
    new_df.loc[df.index[high_indices], "HighLow"] = 1.0
    new_df.loc[df.index[high_indices], "Level"] = swing_highs['High'].values

    swing_lows = df.iloc[low_indices][['Low']].copy()
    new_df.loc[df.index[low_indices], "HighLow"] = -1.0
    new_df.loc[df.index[low_indices], "Level"] = swing_lows['Low'].values

    
    return new_df, atr

def build_market_structure(df: pd.DataFrame, prominence_factor: float = 7.5) -> List[Dict]:
    """
    Builds and classifies the market structure (HH, LH, HL, LL) chronologically.
    """
    swing_highs, swing_lows = detect_swing_points_scipy(df, prominence_factor)
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

def detect_swing_points_by_retracement(ohlc: pd.DataFrame, swing_length: int = 50, minimum_retracement: float = 50.0
    ) -> pd.DataFrame:
        highs = ohlc["High"].to_numpy()
        lows = ohlc["Low"].to_numpy()
        length = len(ohlc)

        highlow = np.full(length, np.nan)
        level = np.full(length, np.nan)
        direction = np.full(length, np.nan)
        current_retracement = np.full(length, np.nan)
        deepest_retracement = np.full(length, np.nan)

        expect = None  # 'low' or 'high'

        for i in range(swing_length, length):
            window_high = highs[i - swing_length : i+1]
            window_low = lows[i - swing_length : i+1]

            if expect is None:
                # Initialisierung: erstes Extremum suchen
                if highs[i] == np.max(window_high):
                    highlow[i] = 1
                    level[i] = highs[i]
                    direction[i] = 1
                    expect = 'low'
                    # print("Initial Swing High at index", i)
                    highlow[0] = -1
                    level[0] = lows[0]
                    direction[0:i] = -1
                elif lows[i] == np.min(window_low):
                    highlow[i] = -1
                    level[i] = lows[i]
                    direction[i] = -1
                    expect = 'high'
                    # print("Initial Swing Low at index", i)
                    highlow[0] = 1
                    level[0] = highs[0]
                    direction[0:i] = 1
                continue
            
            swing_highs_lows = np.where(~np.isnan(highlow))[0]
            if len(swing_highs_lows) < 2:
                continue

            if highs[i] == np.max(window_high):
                if expect == 'high':
                    current_retracement[i] = round(100 - ((highs[i] - level[swing_highs_lows[-2]]) / 
                                  (level[swing_highs_lows[-1]] - level[swing_highs_lows[-2]])) * 100, 2)
                    deepest_retracement[i] = max(deepest_retracement[i - 1], current_retracement[i])
                    if current_retracement[i] >= minimum_retracement:
                        highlow[i] = 1
                        level[i] = highs[i]
                        direction[i] = 1
                        expect = 'low'
                    else:
                        direction[i] = -1
                                               

                elif expect == 'low':
                    if highs[i] >= level[swing_highs_lows[-1]]:
                        highlow[swing_highs_lows[-1]] = np.nan
                        level[swing_highs_lows[-1]] = np.nan
                        highlow[i] = 1
                        level[i] = highs[i]
                    direction[i] = 1
                    swing_highs_lows = np.where(~np.isnan(highlow))[0]
                    if len(swing_highs_lows) < 3:
                        continue
                    current_retracement[i] = round(100 - ((highs[i] - level[swing_highs_lows[-3]]) / 
                                                    (level[swing_highs_lows[-2]] - level[swing_highs_lows[-3]])) * 100, 2)
                    deepest_retracement[i] = 0.0 #if deepest_retracement[i - 1] is np.nan else deepest_retracement[i - 1]


            elif lows[i] == np.min(window_low):
                if expect == 'low':
                    current_retracement[i] = round(100 - ((lows[i] - level[swing_highs_lows[-2]]) / 
                                  (level[swing_highs_lows[-1]] - level[swing_highs_lows[-2]])) * 100, 2)
                    deepest_retracement[i] = max(deepest_retracement[i - 1], current_retracement[i])
                    if current_retracement[i] >= minimum_retracement:
                        highlow[i] = -1
                        level[i] = lows[i]
                        direction[i] = -1
                        expect = 'high'
                    else:
                        direction[i] = 1

                elif expect == 'high':
                    if lows[i] <= level[swing_highs_lows[-1]]:
                        highlow[swing_highs_lows[-1]] = np.nan
                        level[swing_highs_lows[-1]] = np.nan
                        highlow[i] = -1
                        level[i] = lows[i]
                    direction[i] = -1
                    swing_highs_lows = np.where(~np.isnan(highlow))[0]
                    if len(swing_highs_lows) < 3:
                        continue
                    current_retracement[i] = round(100 - ((lows[i] - level[swing_highs_lows[-3]]) / 
                                                    (level[swing_highs_lows[-2]] - level[swing_highs_lows[-3]])) * 100, 2)
                    deepest_retracement[i] = 0.0 #if deepest_retracement[i - 1] is np.nan else deepest_retracement[i - 1]

            else:
                direction[i] = direction[i - 1]
                if expect == 'high':
                    current_retracement[i] = round(100 - ((highs[i] - level[swing_highs_lows[-2]]) / 
                                                    (level[swing_highs_lows[-1]] - level[swing_highs_lows[-2]])) * 100, 2)
                elif expect == 'low':
                    current_retracement[i] = round(100 - ((lows[i] - level[swing_highs_lows[-2]]) / 
                                                    (level[swing_highs_lows[-1]] - level[swing_highs_lows[-2]])) * 100, 2)
                deepest_retracement[i] = max(deepest_retracement[i - 1], current_retracement[i])

        print(deepest_retracement)
        return pd.DataFrame({
            "HighLow": highlow,
            "Level": level,
            #"Direction": direction,
            "Retracement": current_retracement,
            #"DeepestRetracement": deepest_retracement
        }, index=ohlc.index)

def do_highlow_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies swing points into HH, LH, HL, LL and adds as a new column.
    """
    classification_series = pd.Series(index=df.index, dtype="object")
    structure = df[['HighLow', 'Level']].dropna()

    last_high: Optional[Dict] = None
    last_low: Optional[Dict] = None

    for rows in structure.itertuples():
        price = rows.Level
        point_type = rows.HighLow
        timestamp = rows.Index
        classification = None
        if point_type == 1:  # point_type == 'high'
            if last_high:
                classification = "HH" if price >= last_high['price'] else "LH"
            last_high = {'timestamp': timestamp, 'price': price}
        elif point_type == -1: # point_type == 'low'
            if last_low:
                classification = "HL" if price >= last_low['price'] else "LL"
            last_low = {'timestamp': timestamp, 'price': price}
        else:
            print("Unexpected HighLow type:", point_type)
        
        if classification:
            classification_series.at[timestamp] = classification


    df.insert(loc=2, column='Classification', value=classification_series) 
    return df

def enforce_alternating_swings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that swing highs and lows alternate (1 → -1 → 1 → ...).
    If multiple consecutive highs or lows appear, only the most extreme one is kept.
    
    Works vectorized for performance.
    """
    df_origin = df
    df = df.sort_index().copy().dropna()

    # Extract arrays for vectorized processing
    types = df['HighLow'].values.astype(float)
    levels = df['Level'].values.astype(float)

    # Identify indices where the swing type changes (1 → -1 or -1 → 1)
    change_mask = np.concatenate(([True], types[1:] != types[:-1]))

    # Group consecutive segments of same type
    group_ids = np.cumsum(change_mask) - 1

    # Aggregate per segment: choose the most extreme level
    # keep_indices = []
    delete_indices = []
    for g in np.unique(group_ids):
        segment_idx = np.where(group_ids == g)[0]
        seg_type = types[segment_idx[0]]

        if seg_type == 1:
            # For highs: keep the highest price
            best_idx = segment_idx[np.argmax(levels[segment_idx])]
            delete_indices.extend(idx for idx in segment_idx if idx != best_idx)
        else:
            # For lows: keep the lowest price
            best_idx = segment_idx[np.argmin(levels[segment_idx])]
            delete_indices.extend(idx for idx in segment_idx if idx != best_idx)
        # keep_indices.append(best_idx)

    # Build filtered DataFrame
    to_clean = df.iloc[delete_indices]
    
    df_origin.loc[to_clean.index, :] = np.nan

    return df_origin


def add_retracements(ohlc: pd.DataFrame, swing_highs_lows: pd.DataFrame, only_swing_points: bool = True) -> pd.DataFrame:
    """
    Retracement
    This method returns the percentage of a retracement from the swing high or low

    parameters:
    swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function

    returns:
    Direction = 1 if bullish retracement, -1 if bearish retracement
    CurrentRetracement = the current retracement percentage from the swing high or low
    DeepestRetracement = the deepest retracement percentage from the swing high or low
    """
    if only_swing_points:
        ohlc = swing_highs_lows.dropna()

    direction = pd.Series(index=ohlc.index, dtype="int")
    current_retracement = pd.Series(index=ohlc.index, dtype="float")
    deepest_retracement = pd.Series(index=ohlc.index, dtype="float")

    top_current = 0
    top_previous = 0
    bottom_current = 0
    bottom_previous = 0

    remove_first_count = 0

    for i, ts in enumerate(ohlc.index):

        if not only_swing_points:
        
            if swing_highs_lows["HighLow"].at[ts] == 1:
                if remove_first_count < 1:
                    top_previous = top_current = swing_highs_lows["Level"].at[ts]
                    remove_first_count += 1
                    direction.iloc[i] = direction.iloc[i - 1] if i > 0 else 0
                else:
                    top_previous, top_current = top_current, swing_highs_lows["Level"].at[ts]  
                    # print( top_previous, top_current)
                    direction.iloc[i] = 1
                
                # deepest_retracement[i] = 0
            elif swing_highs_lows["HighLow"].at[ts] == -1:
                if remove_first_count < 1:
                    bottom_previous = bottom_current = swing_highs_lows["Level"].at[ts]
                    remove_first_count += 1
                    direction.iloc[i] = direction.iloc[i - 1] if i > 0 else 0
                else:
                    bottom_previous, bottom_current = bottom_current, swing_highs_lows["Level"].at[ts]
                    # print( bottom_previous, bottom_current)
                    direction.iloc[i] = -1
                # deepest_retracement[i] = 0
            else:
                direction.iloc[i] = direction.iloc[i - 1] if i > 0 else 0

            if direction.iloc[i - 1] == 1:
                if direction.iloc[i] == 1:
                    current_retracement.iloc[i] = round(
                        100 - (((ohlc["Low"].at[ts] - bottom_current) / (top_current - bottom_current)) * 100), 1
                    )
                else:
                    current_retracement.iloc[i] = round(
                        100 - (((ohlc["Low"].at[ts] - bottom_previous) / (top_current - bottom_previous)) * 100), 1
                    )
                deepest_retracement.iloc[i] = max(
                    (
                        deepest_retracement.iloc[i - 1]
                        if i > 0 and direction.iloc[i - 2] == 1
                        else 0
                    ),
                    current_retracement.iloc[i],
                )
            if direction.iloc[i - 1] == -1:
                if direction.iloc[i] == -1:
                    current_retracement.iloc[i] = round(
                        100 - ((ohlc["High"].at[ts] - top_current) / (bottom_current - top_current)) * 100, 1
                    )
                else:
                    current_retracement.iloc[i] = round(
                        100 - ((ohlc["High"].at[ts] - top_previous) / (bottom_current - top_previous)) * 100, 1
                    )

                deepest_retracement.iloc[i] = max(
                    (
                        deepest_retracement.iloc[i - 1]
                        if i > 0 and direction.iloc[i - 2] == -1
                        else 0
                    ),
                    current_retracement.iloc[i],
                )
        else:
            if swing_highs_lows["HighLow"].at[ts] == 1:
                top_previous, top_current = top_current, swing_highs_lows["Level"].at[ts]  
                direction.iloc[i] = 1
            elif swing_highs_lows["HighLow"].at[ts] == -1:
                bottom_previous, bottom_current = bottom_current, swing_highs_lows["Level"].at[ts]
                direction.iloc[i] = -1

            if direction.iloc[i - 1] == 1:
                current_retracement.iloc[i] = round(
                    100 - (((bottom_current - bottom_previous) / (top_current - bottom_previous)) * 100), 1
                )
                deepest_retracement.iloc[i] = max(
                    (
                        deepest_retracement.iloc[i - 1]
                        if i > 0 and direction.iloc[i - 2] == 1
                        else 0
                    ),
                    current_retracement.iloc[i],
                )
            if direction.iloc[i - 1] == -1:
                current_retracement.iloc[i] = round(
                    100 - ((top_current - top_previous) / (bottom_current - top_previous)) * 100, 1
                )

                deepest_retracement.iloc[i] = max(
                    (
                        deepest_retracement.iloc[i - 1]
                        if i > 0 and direction.iloc[i - 2] == -1
                        else 0
                    ),
                    current_retracement.iloc[i],
                )

    #direction = pd.Series(direction, name="Direction")
    current_retracement = pd.Series(current_retracement, name="CurrentRetracement")
    deepest_retracement = pd.Series(deepest_retracement, name="DeepestRetracement")
    #swing_highs_lows['Direction'] = direction
    swing_highs_lows['Retracement'] = current_retracement
    # swing_highs_lows['DeepestRetracement'] = deepest_retracement

    return swing_highs_lows

def detect_trend(swing_points: pd.DataFrame, trend_window: int = 4) -> str:
    """
    Detects the current trend based on the last N swing points.
    """
    if len(swing_points) < trend_window:
        print("Not enough swing points to determine trend.")
        return "sideways"

    recent_swings = swing_points.dropna().iloc[-trend_window:]
    types = recent_swings['Classification'].tolist()

    is_uptrend = "HH" in types and "HL" in types and "LL" not in types
    is_downtrend = "LL" in types and "LH" in types and "HH" not in types

    if is_uptrend:
        return "uptrend"
    elif is_downtrend:
        return "downtrend"
    else:
        return "sideways"

def get_market_analysis(df: pd.DataFrame, prominence_factor: float = 7.5, trend_window: int = 10) -> Dict:
    """
    Main function to get market structure and confirm the current trend.
    """
    structure = build_market_structure(df, prominence_factor)
    
    trend = "sideways"
    if len(structure) >= trend_window:
        recent_structure = structure[-trend_window:]
        types = [p['type'] for p in recent_structure]
        
        is_uptrend = "HH" in types and "HL" in types and "LL" not in types
        is_downtrend = "LL" in types and "LH" in types and "HH" not in types

        if is_uptrend:
            trend = "uptrend"
        elif is_downtrend:
            trend = "downtrend"

    swing_highs, swing_lows = detect_swing_points_scipy(df, prominence_factor)
    
    return {
        "trend": trend,
        "structure": structure,
        "swing_highs": list(swing_highs.reset_index().to_records(index=False)),
        "swing_lows": list(swing_lows.reset_index().to_records(index=False))
    }

def analyse_market_structure(df: pd.DataFrame, swing_length: int = 10, trend_window: int = 4, prominence_factor: float = 7.5, based_on_atr = True) -> Tuple[pd.DataFrame, str]:
    """
    Main function to get market structure and confirm the current trend.
    """
    if not based_on_atr:
        swing_highs_lows = detect_swing_points_by_retracement(df, swing_length=swing_length, minimum_retracement=50.0)
    else:
        swing_highs_lows, _ = detect_swing_points_by_atr(df, prominence_factor=prominence_factor)
    
    swing_highs_lows = enforce_alternating_swings(swing_highs_lows)
    swing_highs_lows = do_highlow_classification(swing_highs_lows)
    
    if not "Direction" in swing_highs_lows.columns or not "CurrentRetracement" in swing_highs_lows.columns or not "DeepestRetracement" in swing_highs_lows.columns:
        swing_highs_lows = add_retracements(df, swing_highs_lows, only_swing_points=True)

    trend = detect_trend(swing_highs_lows, trend_window=trend_window)

    return swing_highs_lows, trend