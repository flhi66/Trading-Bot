import pandas as pd
from typing import List, Dict, Literal, Optional

# --- Type Definitions for Clarity ---
StructurePoint = Dict[str, pd.Timestamp | str | float]
TrendType = Literal["uptrend", "downtrend", "sideways"]
EventType = Literal["BOS", "CHOCH"]

# --- Helper Function ---
def find_last_swing(
    structure: List[StructurePoint],
    swing_type: Literal["HH", "HL", "LH", "LL"],
    before_index: int
) -> Optional[StructurePoint]:
    """Finds the most recent occurrence of a given swing type before a certain index."""
    for i in range(before_index - 1, -1, -1):
        if structure[i]["type"] == swing_type:
            return structure[i]
    return None

# --- Main Detection Logic ---
def get_market_events(structure: List[StructurePoint]) -> List[Dict]:
    """
    Analyzes the market structure to detect Break of Structure (BOS) and Change of Character (CHOCH) events.
    """
    if len(structure) < 2:
        return []

    events = []
    
    for i in range(1, len(structure)):
        current_swing = structure[i]
        
        # === 1. Bullish Break of Structure (BOS) ===
        # A new HH breaks the previous HH.
        if current_swing["type"] == "HH":
            prev_hh = find_last_swing(structure, "HH", i)
            if prev_hh and current_swing["price"] > prev_hh["price"]:
                confirmed_low = find_last_swing(structure, "HL", i)
                if confirmed_low:
                    events.append({
                        "type": "BOS",
                        "direction": "Bullish",
                        "timestamp": current_swing["timestamp"],
                        "price": current_swing["price"],
                        "tjl1": {"timestamp": prev_hh["timestamp"], "price": prev_hh["price"]},
                        "tjl2_a_plus": {"timestamp": confirmed_low["timestamp"], "price": confirmed_low["price"]}
                    })

        # === 2. Bearish Break of Structure (BOS) ===
        # A new LL breaks the previous LL.
        elif current_swing["type"] == "LL":
            prev_ll = find_last_swing(structure, "LL", i)
            if prev_ll and current_swing["price"] < prev_ll["price"]:
                confirmed_high = find_last_swing(structure, "LH", i)
                if confirmed_high:
                    events.append({
                        "type": "BOS",
                        "direction": "Bearish",
                        "timestamp": current_swing["timestamp"],
                        "price": current_swing["price"],
                        "tjl1": {"timestamp": prev_ll["timestamp"], "price": prev_ll["price"]},
                        "tjl2_a_plus": {"timestamp": confirmed_high["timestamp"], "price": confirmed_high["price"]}
                    })

        # === 3. Bearish Change of Character (CHOCH) ===
        # A new LL breaks the last significant HL after an uptrend move.
        if current_swing["type"] == "LL":
            last_hl = find_last_swing(structure, "HL", i)
            if last_hl and current_swing["price"] < last_hl["price"]:
                # Check if this HL was part of a recent uptrend structure
                qml_level = find_last_swing(structure, "HH", i)
                if qml_level and qml_level["timestamp"] > last_hl["timestamp"]:
                    events.append({
                        "type": "CHOCH",
                        "direction": "Bearish", # A break of bullish structure
                        "timestamp": current_swing["timestamp"],
                        "price": current_swing["price"],
                        "sbr_level": {"timestamp": last_hl["timestamp"], "price": last_hl["price"]},
                        "qml_a_plus": {"timestamp": qml_level["timestamp"], "price": qml_level["price"]},
                    })

        # === 4. Bullish Change of Character (CHOCH) ===
        # A new HH breaks the last significant LH after a downtrend move.
        elif current_swing["type"] == "HH":
            last_lh = find_last_swing(structure, "LH", i)
            if last_lh and current_swing["price"] > last_lh["price"]:
                # Check if this LH was part of a recent downtrend structure
                qml_level = find_last_swing(structure, "LL", i)
                if qml_level and qml_level["timestamp"] > last_lh["timestamp"]:
                    events.append({
                        "type": "CHOCH",
                        "direction": "Bullish", # A break of bearish structure
                        "timestamp": current_swing["timestamp"],
                        "price": current_swing["price"],
                        "rbs_level": {"timestamp": last_lh["timestamp"], "price": last_lh["price"]},
                        "qml_a_plus": {"timestamp": qml_level["timestamp"], "price": qml_level["price"]},
                    })
                    
    return events