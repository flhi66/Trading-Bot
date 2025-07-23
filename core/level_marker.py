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

        # === FOR UPTREND ===
        if trend == "uptrend":
            if current["type"] == "HH":
                if prev_hh and current["price"] > prev_hh["price"]:
                    bos_confirmed = True
                    last_bos_idx = i
                    bos_list.append({
                        "type": "BOS",
                        "trend": "uptrend",
                        "timestamp": current["timestamp"],
                        "price": current["price"]
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
                            "price": prev_hl["price"] * 0.98  # support becomes resistance
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
                    bos_list.append({
                        "type": "BOS",
                        "trend": "downtrend",
                        "timestamp": current["timestamp"],
                        "price": current["price"]
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
                            "price": prev_lh["price"] * 1.02  # resistance becomes support
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
