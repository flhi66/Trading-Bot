import pandas as pd
from typing import List, Dict, Literal, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# --- Enhanced Type Definitions ---
class SwingType(Enum):
    HH = "HH"  # Higher High
    HL = "HL"  # Higher Low
    LH = "LH"  # Lower High
    LL = "LL"  # Lower Low

class TrendState(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"

class EventType(Enum):
    BOS = "BOS"
    CHOCH = "CHOCH"

@dataclass
class StructurePoint:
    timestamp: pd.Timestamp
    price: float
    swing_type: SwingType
    index: int

@dataclass
class MarketEvent:
    event_type: EventType
    direction: Literal["Bullish", "Bearish"]
    timestamp: pd.Timestamp
    price: float
    confidence: float  # 0-1 confidence score
    broken_level: Dict
    context: Dict
    description: str

class MarketStructureAnalyzer:
    def __init__(self, min_structure_points: int = 3, confidence_threshold: float = 0.6):
        self.min_structure_points = min_structure_points
        self.confidence_threshold = confidence_threshold
        self.current_trend = TrendState.UNKNOWN
        
    def find_last_swing(self, structure: List[StructurePoint], swing_type: SwingType, before_index: int) -> Optional[StructurePoint]:
        """Find the most recent swing of specified type before given index."""
        for i in range(before_index - 1, -1, -1):
            if structure[i].swing_type == swing_type:
                return structure[i]
        return None
    
    def get_trend_state(self, structure: List[StructurePoint], current_index: int) -> TrendState:
        """Determine current trend state based on recent structure."""
        if current_index < 2:
            return TrendState.UNKNOWN
            
        # Look at last 3-4 swings to determine trend
        recent_swings = structure[max(0, current_index-3):current_index+1]
        
        hh_count = sum(1 for s in recent_swings if s.swing_type == SwingType.HH)
        hl_count = sum(1 for s in recent_swings if s.swing_type == SwingType.HL)
        lh_count = sum(1 for s in recent_swings if s.swing_type == SwingType.LH)
        ll_count = sum(1 for s in recent_swings if s.swing_type == SwingType.LL)
        
        # Uptrend: More HH/HL than LH/LL
        if (hh_count + hl_count) > (lh_count + ll_count) and hh_count > 0:
            return TrendState.UPTREND
        # Downtrend: More LH/LL than HH/HL
        elif (lh_count + ll_count) > (hh_count + hl_count) and ll_count > 0:
            return TrendState.DOWNTREND
        else:
            return TrendState.SIDEWAYS
    
    def calculate_confidence(self, event_type: EventType, price_break: float, volume_context: Optional[Dict] = None) -> float:
        """Calculate confidence score for the detected event."""
        base_confidence = 0.7
        
        # Adjust based on price break magnitude
        if abs(price_break) > 10:  # Significant break
            base_confidence += 0.2
        elif abs(price_break) < 2:  # Weak break
            base_confidence -= 0.2
            
        # Add volume confirmation if available
        if volume_context and volume_context.get('above_average', False):
            base_confidence += 0.1
            
        return min(1.0, max(0.1, base_confidence))
    
    def detect_bos_events(self, structure: List[StructurePoint]) -> List[MarketEvent]:
        """Detect Break of Structure events."""
        events = []
        
        for i in range(1, len(structure)):
            current = structure[i]
            
            # Bullish BOS: New HH breaks previous significant HH
            if current.swing_type == SwingType.HH:
                prev_hh = self.find_last_swing(structure, SwingType.HH, i)
                if prev_hh and current.price > prev_hh.price:
                    # Ensure there's a confirmed low between them
                    intermediate_low = self.find_last_swing(structure, SwingType.HL, i)
                    if intermediate_low and intermediate_low.timestamp > prev_hh.timestamp:
                        confidence = self.calculate_confidence(EventType.BOS, current.price - prev_hh.price)
                        if confidence >= self.confidence_threshold:
                            events.append(MarketEvent(
                                event_type=EventType.BOS,
                                direction="Bullish",
                                timestamp=current.timestamp,
                                price=current.price,
                                confidence=confidence,
                                broken_level={"timestamp": prev_hh.timestamp, "price": prev_hh.price},
                                context={"intermediate_low": {"timestamp": intermediate_low.timestamp, "price": intermediate_low.price}},
                                description=f"Bullish BOS: New HH at {current.price:.2f} breaks previous HH at {prev_hh.price:.2f}"
                            ))
            
            # Bearish BOS: New LL breaks previous significant LL
            elif current.swing_type == SwingType.LL:
                prev_ll = self.find_last_swing(structure, SwingType.LL, i)
                if prev_ll and current.price < prev_ll.price:
                    # Ensure there's a confirmed high between them
                    intermediate_high = self.find_last_swing(structure, SwingType.LH, i)
                    if intermediate_high and intermediate_high.timestamp > prev_ll.timestamp:
                        confidence = self.calculate_confidence(EventType.BOS, prev_ll.price - current.price)
                        if confidence >= self.confidence_threshold:
                            events.append(MarketEvent(
                                event_type=EventType.BOS,
                                direction="Bearish",
                                timestamp=current.timestamp,
                                price=current.price,
                                confidence=confidence,
                                broken_level={"timestamp": prev_ll.timestamp, "price": prev_ll.price},
                                context={"intermediate_high": {"timestamp": intermediate_high.timestamp, "price": intermediate_high.price}},
                                description=f"Bearish BOS: New LL at {current.price:.2f} breaks previous LL at {prev_ll.price:.2f}"
                            ))
        
        return events
    
    def detect_choch_events(self, structure: List[StructurePoint]) -> List[MarketEvent]:
        """Detect Change of Character events."""
        events = []
        
        for i in range(2, len(structure)):  # Need at least 3 points for CHOCH
            current = structure[i]
            current_trend = self.get_trend_state(structure, i-1)
            
            # Bearish CHOCH: LL breaks HL during uptrend (trend change)
            if current.swing_type == SwingType.LL and current_trend == TrendState.UPTREND:
                last_hl = self.find_last_swing(structure, SwingType.HL, i)
                if last_hl and current.price < last_hl.price:
                    # Verify this was actually an uptrend by checking for recent HH
                    recent_hh = self.find_last_swing(structure, SwingType.HH, i)
                    if recent_hh and recent_hh.timestamp > last_hl.timestamp:
                        confidence = self.calculate_confidence(EventType.CHOCH, last_hl.price - current.price)
                        if confidence >= self.confidence_threshold:
                            events.append(MarketEvent(
                                event_type=EventType.CHOCH,
                                direction="Bearish",
                                timestamp=current.timestamp,
                                price=current.price,
                                confidence=confidence,
                                broken_level={"timestamp": last_hl.timestamp, "price": last_hl.price},
                                context={"trend_before": "uptrend", "confirming_hh": {"timestamp": recent_hh.timestamp, "price": recent_hh.price}},
                                description=f"Bearish CHOCH: LL at {current.price:.2f} breaks HL at {last_hl.price:.2f}, trend change from uptrend"
                            ))
            
            # Bullish CHOCH: HH breaks LH during downtrend (trend change)
            elif current.swing_type == SwingType.HH and current_trend == TrendState.DOWNTREND:
                last_lh = self.find_last_swing(structure, SwingType.LH, i)
                if last_lh and current.price > last_lh.price:
                    # Verify this was actually a downtrend by checking for recent LL
                    recent_ll = self.find_last_swing(structure, SwingType.LL, i)
                    if recent_ll and recent_ll.timestamp > last_lh.timestamp:
                        confidence = self.calculate_confidence(EventType.CHOCH, current.price - last_lh.price)
                        if confidence >= self.confidence_threshold:
                            events.append(MarketEvent(
                                event_type=EventType.CHOCH,
                                direction="Bullish",
                                timestamp=current.timestamp,
                                price=current.price,
                                confidence=confidence,
                                broken_level={"timestamp": last_lh.timestamp, "price": last_lh.price},
                                context={"trend_before": "downtrend", "confirming_ll": {"timestamp": recent_ll.timestamp, "price": recent_ll.price}},
                                description=f"Bullish CHOCH: HH at {current.price:.2f} breaks LH at {last_lh.price:.2f}, trend change from downtrend"
                            ))
        
        return events
    
    def get_market_events(self, structure_data: List[Dict]) -> List[MarketEvent]:
        """
        Main function to analyze market structure and detect BOS/CHOCH events.
        
        Args:
            structure_data: List of dicts with keys: timestamp, price, type, (optional: index)
        """
        if len(structure_data) < self.min_structure_points:
            return []
        
        # Convert to structured format
        structure = []
        for i, point in enumerate(structure_data):
            structure.append(StructurePoint(
                timestamp=point["timestamp"],
                price=point["price"],
                swing_type=SwingType(point["type"]),
                index=point.get("index", i)
            ))
        
        # Detect BOS and CHOCH events separately to avoid duplicates
        bos_events = self.detect_bos_events(structure)
        choch_events = self.detect_choch_events(structure)
        
        # Combine and remove any potential duplicates at same timestamp
        all_events = bos_events + choch_events
        unique_events = []
        seen_timestamps = set()
        
        # Sort by timestamp and remove duplicates (prefer higher confidence)
        all_events.sort(key=lambda x: (x.timestamp, -x.confidence))
        for event in all_events:
            if event.timestamp not in seen_timestamps:
                unique_events.append(event)
                seen_timestamps.add(event.timestamp)
        
        return unique_events

# --- Usage Example ---
def example_usage():
    # Sample structure data
    sample_structure = [
        {"timestamp": pd.Timestamp("2024-01-01 10:00"), "price": 3300.0, "type": "LL"},
        {"timestamp": pd.Timestamp("2024-01-01 11:00"), "price": 3320.0, "type": "LH"},
        {"timestamp": pd.Timestamp("2024-01-01 12:00"), "price": 3290.0, "type": "LL"},
        {"timestamp": pd.Timestamp("2024-01-01 13:00"), "price": 3330.0, "type": "HH"},  # Potential CHOCH
        {"timestamp": pd.Timestamp("2024-01-01 14:00"), "price": 3340.0, "type": "HH"},  # Potential BOS
    ]
    
    analyzer = MarketStructureAnalyzer(confidence_threshold=0.6)
    events = analyzer.get_market_events(sample_structure)
    
    for event in events:
        print(f"{event.event_type.value} - {event.direction} @ {event.price:.2f} (Confidence: {event.confidence:.2f})")
        print(f"Description: {event.description}")
        print("---")

if __name__ == "__main__":
    example_usage()