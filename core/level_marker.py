from typing import Literal, TypedDict

StructurePoint = dict  # Expected: {"timestamp": ..., "type": "HH"/"HL"/"LH"/"LL", "price": float}
TrendType = Literal["uptrend", "downtrend", "sideways"]

def detect_bos_choch(structure: list[StructurePoint], trend: TrendType):
    bos_list = []
    choch_list = []

    prev_hh = None
    prev_hl = None
    prev_lh = None
    prev_ll = None
    bos_confirmed = False
    last_bos_idx = None

    for i in range(2, len(structure)):
        current = structure[i]
        previous = structure[i - 1]

        # === FOR UPTREND ===
        if trend == "uptrend":
            if current["type"] == "HH":
                if prev_hh and current["price"] > prev_hh["price"]:
                    bos_confirmed = True
                    last_bos_idx = i

                    # TJL1: Last candle of previous HH (upper wick and 2% body buffer)
                    tjl1 = {
                        "timestamp": prev_hh["timestamp"],
                        "price": prev_hh["price"] * 1.02  # Add buffer to upper wick
                    }

                    # TJL2: Last candle of previous HL
                    tjl2 = {
                        "timestamp": prev_hl["timestamp"],
                        "price": prev_hl["price"] * 0.98  # Slight buffer to lower side
                    } if prev_hl else None

                    bos_list.append({
                        "type": "BOS",
                        "trend": "uptrend",
                        "timestamp": current["timestamp"],
                        "price": current["price"],
                        "direction": "bullish",
                        "tjl1": tjl1,
                        "tjl2": tjl2
                    })
                prev_hh = current

            elif current["type"] == "HL":
                if bos_confirmed and prev_hl and current["price"] < prev_hl["price"]:
                    qml_candle = structure[last_bos_idx - 1]
                    choch_list.append({
                        "type": "CHOCH",
                        "direction": "bearish",
                        "timestamp": current["timestamp"],
                        "price": current["price"],
                        "qml_level": {
                            "timestamp": qml_candle["timestamp"],
                            "price": qml_candle["price"]
                        },
                        "sbr_level": {
                            "timestamp": prev_hl["timestamp"],
                            "price": prev_hl["price"] * 0.98
                        },
                        "dt_level": {
                            "timestamp": current["timestamp"],
                            "price": current["price"]
                        }
                    })
                    bos_confirmed = False
                prev_hl = current

        # === FOR DOWNTREND ===
        elif trend == "downtrend":
            if current["type"] == "LL":
                if prev_ll and current["price"] < prev_ll["price"]:
                    bos_confirmed = True
                    last_bos_idx = i

                    tjl1 = {
                        "timestamp": prev_ll["timestamp"],
                        "price": prev_ll["price"] * 0.98
                    }

                    tjl2 = {
                        "timestamp": prev_lh["timestamp"],
                        "price": prev_lh["price"] * 1.02
                    } if prev_lh else None

                    bos_list.append({
                        "type": "BOS",
                        "trend": "downtrend",
                        "timestamp": current["timestamp"],
                        "price": current["price"],
                        "direction": "bearish",
                        "tjl1": tjl1,
                        "tjl2": tjl2
                    })
                prev_ll = current

            elif current["type"] == "LH":
                if bos_confirmed and prev_lh and current["price"] > prev_lh["price"]:
                    qml_candle = structure[last_bos_idx - 1]
                    choch_list.append({
                        "type": "CHOCH",
                        "direction": "bullish",
                        "timestamp": current["timestamp"],
                        "price": current["price"],
                        "qml_level": {
                            "timestamp": qml_candle["timestamp"],
                            "price": qml_candle["price"]
                        },
                        "rbs_level": {
                            "timestamp": prev_lh["timestamp"],
                            "price": prev_lh["price"] * 1.02
                        },
                        "db_level": {
                            "timestamp": current["timestamp"],
                            "price": current["price"]
                        }
                    })
                    bos_confirmed = False
                prev_lh = current

    return bos_list, choch_list


def mark_bos_choch_levels(structure: list[StructurePoint], trend: TrendType) -> dict:
    """
    Returns:
        {
            "bos": [ { type, trend, timestamp, price }, ... ],
            "choch": [ { type, direction, timestamp, price, qml_level, rbs/sbr_level, db/dt_level }, ... ]
        }
    """
    bos_list, choch_list = detect_bos_choch(structure, trend)
    return {
        "bos": bos_list,
        "choch": choch_list
    }
