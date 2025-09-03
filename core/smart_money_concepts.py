import pandas as pd
from typing import List, Dict, Literal, Optional
from dataclasses import dataclass
from enum import Enum

# --- Data Structures for Clarity ---
class SwingType(Enum):
    HH = "HH"
    HL = "HL"
    LH = "LH"
    LL = "LL"

class EventType(Enum):
    BOS = "BOS"
    CHOCH = "CHOCH"

@dataclass
class StructurePoint:
    timestamp: pd.Timestamp
    price: float
    swing_type: SwingType

@dataclass
class MarketEvent:
    event_type: EventType
    direction: Literal["Bullish", "Bearish"]
    timestamp: pd.Timestamp
    price: float
    confidence: float
    broken_level: Dict
    context: Dict
    description: str

# --- The Enhanced Logic Engine ---
class MarketStructureAnalyzer:
    def __init__(self, confidence_threshold: float = 0.5, lookback_period: int = 5, 
                 choch_confidence_threshold: float = 0.65, min_structure_width: float = 0.0010,
                 min_time_width_hours: float = 2.0):
        self.confidence_threshold = confidence_threshold
        self.lookback_period = lookback_period
        self.choch_confidence_threshold = choch_confidence_threshold  # Configurable CHOCH threshold
        self.min_structure_width = min_structure_width  # Minimum price structure width
        self.min_time_width_hours = min_time_width_hours  # Minimum time structure width

    def _find_last_swing(self, structure: List[StructurePoint], swing_type: SwingType, before_index: int) -> Optional[StructurePoint]:
        """Find the most recent swing of specified type before given index"""
        for i in range(before_index - 1, -1, -1):
            if structure[i].swing_type == swing_type:
                return structure[i]
        return None

    def _find_all_swings(self, structure: List[StructurePoint], swing_type: SwingType, before_index: int, limit: int = 3) -> List[StructurePoint]:
        """Find multiple swings of specified type for better pattern recognition"""
        swings = []
        for i in range(before_index - 1, -1, -1):
            if structure[i].swing_type == swing_type:
                swings.append(structure[i])
                if len(swings) >= limit:
                    break
        return swings

    def _get_trend_state(self, structure: List[StructurePoint], current_index: int) -> Literal["uptrend", "downtrend", "sideways"]:
        """Enhanced trend detection with better pattern recognition and CHOCH responsiveness"""
        if current_index < 3:
            return "sideways"
        
        # Look at recent swings - use a more flexible lookback
        lookback = min(8, current_index)  # Increased lookback for better trend detection
        recent_swings = structure[max(0, current_index - lookback):current_index]
        
        if len(recent_swings) < 2:
            return "sideways"
        
        # Check for recent CHOCH events that should override trend calculation
        # Look for recent LL breaking below HL (bearish CHOCH) or HH breaking above LH (bullish CHOCH)
        for i in range(len(recent_swings) - 1, 0, -1):
            current_swing = recent_swings[i]
            prev_swing = recent_swings[i-1]
            
            # Bearish CHOCH: LL breaking below HL
            if (current_swing.swing_type == SwingType.LL and 
                prev_swing.swing_type == SwingType.HL and 
                current_swing.price < prev_swing.price):
                # Recent bearish CHOCH detected - trend should be downtrend
                return "downtrend"
            
            # Bullish CHOCH: HH breaking above LH
            if (current_swing.swing_type == SwingType.HH and 
                prev_swing.swing_type == SwingType.LH and 
                current_swing.price > prev_swing.price):
                # Recent bullish CHOCH detected - trend should be uptrend
                return "uptrend"
        
        # Count swing patterns in recent history
        hh_count = sum(1 for s in recent_swings if s.swing_type == SwingType.HH)
        hl_count = sum(1 for s in recent_swings if s.swing_type == SwingType.HL)
        ll_count = sum(1 for s in recent_swings if s.swing_type == SwingType.LL)
        lh_count = sum(1 for s in recent_swings if s.swing_type == SwingType.LH)
        
        # More flexible trend logic
        bullish_signals = hh_count + hl_count
        bearish_signals = ll_count + lh_count
        
        # Check for clear uptrend patterns
        if (hh_count > 0 and hl_count > 0) or (bullish_signals > bearish_signals and hh_count > 0):
            return "uptrend"
        # Check for clear downtrend patterns  
        elif (ll_count > 0 and lh_count > 0) or (bearish_signals > bullish_signals and ll_count > 0):
            return "downtrend"
        # Check recent sequence for immediate trend
        elif len(recent_swings) >= 2:
            last_two = [s.swing_type for s in recent_swings[-2:]]
            if SwingType.HH in last_two or SwingType.HL in last_two:
                return "uptrend"
            elif SwingType.LL in last_two or SwingType.LH in last_two:
                return "downtrend"
        
        return "sideways"

    def _calculate_confidence(self, current_point: StructurePoint, broken_level: StructurePoint, 
                            intermediate_point: Optional[StructurePoint] = None, event_type: EventType = None) -> float:
        """Enhanced confidence calculation with event-specific criteria and structure quality checks"""
        base_confidence = 0.6
        
        # Price break strength
        price_break = abs(current_point.price - broken_level.price)
        if price_break > 50:  # Significant break (adjust based on instrument)
            base_confidence += 0.2
        elif price_break < 10:  # Weak break
            base_confidence -= 0.2
        
        # Time factor - more recent breaks are more reliable
        time_diff = (current_point.timestamp - broken_level.timestamp).total_seconds() / 3600  # hours
        if time_diff < 24:  # Within 24 hours
            base_confidence += 0.1
        elif time_diff > 168:  # More than a week
            base_confidence -= 0.1
        
        # Structure integrity and quality check
        if intermediate_point:
            # Check if intermediate point properly separates the levels
            if current_point.swing_type in [SwingType.HH, SwingType.LL]:
                structure_integrity = abs(intermediate_point.price - broken_level.price) / abs(current_point.price - broken_level.price)
                if structure_integrity > 0.3:  # Good separation
                    base_confidence += 0.1
                
                # Check minimum structure width to avoid noise
                structure_width = abs(intermediate_point.price - broken_level.price)
                if structure_width < self.min_structure_width:
                    base_confidence -= 0.3  # Penalize tight structures (noise)
                elif structure_width > self.min_structure_width * 3:
                    base_confidence += 0.1  # Reward wide, meaningful structures
        
        # Event-specific confidence adjustments
        if event_type == EventType.CHOCH:
            # CHOCH requires higher confidence - deeper retracements and stronger reversals
            if price_break < 15:  # Weak CHOCH break
                base_confidence -= 0.15
            elif price_break > 50:  # Strong CHOCH break
                base_confidence += 0.2
            
            # Check for consolidation before CHOCH (price should have moved away from broken level)
            if intermediate_point:
                consolidation_distance = abs(intermediate_point.price - broken_level.price)
                if consolidation_distance > price_break * 0.3:  # Good consolidation
                    base_confidence += 0.15
        
        elif event_type == EventType.BOS:
            # BOS should have clear momentum continuation
            if price_break < 10:  # Weak BOS break
                base_confidence -= 0.1
            elif price_break > 50:  # Strong BOS break
                base_confidence += 0.15
            
            # Bonus for deep retracement (HL very close to LL)
            if intermediate_point and current_point.swing_type == SwingType.HH:
                retracement_depth = abs(intermediate_point.price - broken_level.price)
                if retracement_depth > price_break * 0.4:  # Deep retracement
                    base_confidence += 0.1
        
        # Calculate quality score for trading performance
        quality_score = 0
        if price_break > 30:  # Strong price break
            quality_score += 1
        if event_type == EventType.CHOCH and intermediate_point:  # Clean QML
            quality_score += 1
        if intermediate_point and abs(intermediate_point.price - broken_level.price) > price_break * 0.3:  # Deep retracement
            quality_score += 1
        if time_diff < 48:  # Recent event
            quality_score += 1
        
        # Normalize quality score (0-4 scale)
        normalized_quality = quality_score / 4.0
        
        return min(1.0, max(0.1, base_confidence)), normalized_quality

    def _validate_bos_pattern(self, current: StructurePoint, previous_extreme: StructurePoint, 
                            intermediate: StructurePoint) -> bool:
        """Enhanced BOS pattern validation with comprehensive structure quality checks"""
        # Ensure proper sequence timing
        if not (previous_extreme.timestamp < intermediate.timestamp < current.timestamp):
            return False
        
        # Comprehensive structure width validation (price + time)
        if not self._validate_structure_width(current, previous_extreme, intermediate):
            return False  # Structure too narrow in price or time
        
        # Validate price relationships
        if current.swing_type == SwingType.HH:
            return (intermediate.swing_type == SwingType.HL and 
                   intermediate.price < previous_extreme.price and
                   current.price > previous_extreme.price)
        elif current.swing_type == SwingType.LL:
            return (intermediate.swing_type == SwingType.LH and 
                   intermediate.price > previous_extreme.price and
                   current.price < previous_extreme.price)
        
        return False

    def _validate_choch_pattern(self, current: StructurePoint, broken_level: StructurePoint, 
                              trend_state: str) -> bool:
        """Enhanced CHOCH pattern validation with comprehensive structure quality checks"""
        # Comprehensive structure width validation (price + time)
        if not self._validate_structure_width(current, broken_level):
            return False  # Structure too narrow in price or time
        
        if trend_state == "uptrend" and current.swing_type == SwingType.LL:
            # For bearish CHOCH: LL breaking below the last HL in uptrend
            return (broken_level.swing_type == SwingType.HL and 
                   current.price < broken_level.price)
        elif trend_state == "downtrend" and current.swing_type == SwingType.HH:
            # For bullish CHOCH: HH breaking above the last LH in downtrend  
            return (broken_level.swing_type == SwingType.LH and 
                   current.price > broken_level.price)
        
        # Additional cases for trend change detection
        elif trend_state == "uptrend" and current.swing_type == SwingType.LH:
            # LH in uptrend can also signal CHOCH if it breaks below previous HL
            return (broken_level.swing_type == SwingType.HL and 
                   current.price < broken_level.price)
        elif trend_state == "downtrend" and current.swing_type == SwingType.HL:
            # HL in downtrend can also signal CHOCH if it breaks above previous LH
            return (broken_level.swing_type == SwingType.LH and 
                   current.price > broken_level.price)
        
        return False

    def _validate_structure_width(self, current: StructurePoint, previous: StructurePoint, 
                                 intermediate: Optional[StructurePoint] = None) -> bool:
        """
        Comprehensive structure width validation using both price and time dimensions.
        This ensures we only detect meaningful structures, not noise.
        """
        # Price-based structure width validation
        price_width = abs(current.price - previous.price)
        if price_width < self.min_structure_width:
            return False  # Price structure too narrow
        
        # Time-based structure width validation
        time_width_hours = abs((current.timestamp - previous.timestamp).total_seconds()) / 3600
        if time_width_hours < self.min_time_width_hours:
            return False  # Time structure too narrow
        
        # If intermediate point exists, validate the full structure
        if intermediate:
            # Check intermediate to previous width
            intermediate_prev_width = abs(intermediate.price - previous.price)
            intermediate_prev_time = abs((intermediate.timestamp - previous.timestamp).total_seconds()) / 3600
            
            # Check intermediate to current width
            intermediate_current_width = abs(current.price - intermediate.price)
            intermediate_current_time = abs((current.timestamp - intermediate.timestamp).total_seconds()) / 3600
            
            # All segments should have meaningful width
            if (intermediate_prev_width < self.min_structure_width * 0.5 or 
                intermediate_current_width < self.min_structure_width * 0.5 or
                intermediate_prev_time < self.min_time_width_hours * 0.5 or
                intermediate_current_time < self.min_time_width_hours * 0.5):
                return False
        
        return True

    def _get_dynamic_choch_threshold(self, current: StructurePoint, broken_level: StructurePoint, 
                                   trend_state: str) -> float:
        """
        Dynamic CHOCH confidence threshold based on market conditions.
        This helps adapt to different market volatility and structure quality.
        """
        base_threshold = self.choch_confidence_threshold
        
        # Calculate price break strength
        price_break = abs(current.price - broken_level.price)
        
        # Adjust threshold based on price break strength
        if price_break < 10:  # Very weak break
            return base_threshold + 0.1  # Make it harder
        elif price_break > 50:  # Strong break
            return base_threshold - 0.05  # Make it easier
        
        # Adjust based on trend state
        if trend_state == "sideways":
            return base_threshold + 0.05  # Sideways markets need stronger signals
        
        return base_threshold

    def _check_retracement_confirmation(self, event: MarketEvent, structure: List[StructurePoint], 
                                      current_index: int) -> bool:
        """
        Check if price has retraced back to the broken level after the event.
        This prevents immediate entries on structure breaks.
        """
        if current_index >= len(structure) - 1:
            return False  # No future data to check
        
        broken_price = event.broken_level["price"]
        tolerance = self.min_structure_width * 0.5  # Half the minimum structure width
        
        # Look at future structure points to see if price retraced
        for i in range(current_index + 1, min(current_index + 10, len(structure))):
            future_point = structure[i]
            
            # Check if price has retraced to the broken level (within tolerance)
            if (future_point.price <= broken_price + tolerance and 
                future_point.price >= broken_price - tolerance):
                return True
        
        return False

    def _check_reversal_confirmation(self, event: MarketEvent, structure: List[StructurePoint], 
                                   current_index: int) -> bool:
        """
        Check for reversal confirmation after retracement.
        This ensures we have a proper reversal setup, not just a structure break.
        """
        if current_index >= len(structure) - 2:
            return False  # Need at least 2 future points
        
        # Look for reversal patterns after the event
        for i in range(current_index + 1, min(current_index + 5, len(structure) - 1)):
            current_point = structure[i]
            next_point = structure[i + 1]
            
            # Check for reversal patterns based on event direction
            if event.direction == "Bullish":
                # Look for bullish reversal: LL followed by HL or HH
                if (current_point.swing_type == SwingType.LL and 
                    next_point.swing_type in [SwingType.HL, SwingType.HH]):
                    return True
            elif event.direction == "Bearish":
                # Look for bearish reversal: HH followed by LH or LL
                if (current_point.swing_type == SwingType.HH and 
                    next_point.swing_type in [SwingType.LH, SwingType.LL]):
                    return True
        
        return False

    def get_market_events(self, structure_data: List[Dict]) -> List[MarketEvent]:
        """Enhanced market event detection with improved pattern recognition"""
        if len(structure_data) < 4:
            return []
        
        structure = [StructurePoint(pd.Timestamp(p["timestamp"]), p["price"], SwingType(p["type"])) 
                    for p in structure_data]
        events = []

        for i in range(2, len(structure)):  # Start from index 2 for better pattern validation
            current = structure[i]
            trend_before = self._get_trend_state(structure, i)
            
            # Debug print (remove in production)
            # print(f"Index {i}: {current.swing_type.value} @ {current.price:.2f} - Trend: {trend_before}")
            
            # --- CHOCH Detection (Priority over BOS) ---
            choch_detected = False
            
            # More aggressive CHOCH detection for trend changes
            if trend_before == "uptrend":
                # Any lower swing breaking previous support in uptrend = CHOCH
                if current.swing_type in [SwingType.LL, SwingType.LH]:
                    # Find the most recent HL (support level in uptrend)
                    last_hl = self._find_last_swing(structure, SwingType.HL, i)
                    if last_hl and current.price < last_hl.price:
                        # This is a CHOCH - trend change from bullish to bearish
                        qml_level = self._find_last_swing(structure, SwingType.HH, i)
                        if qml_level:
                            confidence, quality_score = self._calculate_confidence(current, last_hl, qml_level, EventType.CHOCH)
                            dynamic_threshold = self._get_dynamic_choch_threshold(current, last_hl, trend_before)
                            if confidence >= dynamic_threshold:  # Dynamic threshold for CHOCH
                                events.append(MarketEvent(
                                    event_type=EventType.CHOCH,
                                    direction="Bearish",
                                    timestamp=current.timestamp,
                                    price=current.price,
                                    confidence=confidence,
                                    broken_level={"name": "SBR", "timestamp": last_hl.timestamp, "price": last_hl.price},
                                    context={
                                        "a_plus_entry": {"name": "QML", "timestamp": qml_level.timestamp, "price": qml_level.price},
                                        "quality_score": quality_score,
                                        "structure_width": abs(qml_level.price - last_hl.price)
                                    },
                                    description=f"Bearish CHOCH: {current.swing_type.value} @ {current.price:.2f} broke uptrend support @ {last_hl.price:.2f}"
                                ))
                                choch_detected = True
            
            elif trend_before == "downtrend":
                # Any higher swing breaking previous resistance in downtrend = CHOCH  
                if current.swing_type in [SwingType.HH, SwingType.HL]:
                    # Find the most recent LH (resistance level in downtrend)
                    last_lh = self._find_last_swing(structure, SwingType.LH, i)
                    if last_lh and current.price > last_lh.price:
                        # This is a CHOCH - trend change from bearish to bullish
                        qml_level = self._find_last_swing(structure, SwingType.LL, i)
                        if qml_level:
                            confidence, quality_score = self._calculate_confidence(current, last_lh, qml_level, EventType.CHOCH)
                            dynamic_threshold = self._get_dynamic_choch_threshold(current, last_lh, trend_before)
                            if confidence >= dynamic_threshold:  # Dynamic threshold for CHOCH
                                events.append(MarketEvent(
                                    event_type=EventType.CHOCH,
                                    direction="Bullish",
                                    timestamp=current.timestamp,
                                    price=current.price,
                                    confidence=confidence,
                                    broken_level={"name": "RBS", "timestamp": last_lh.timestamp, "price": last_lh.price},
                                    context={
                                        "a_plus_entry": {"name": "QML", "timestamp": qml_level.timestamp, "price": qml_level.price},
                                        "quality_score": quality_score,
                                        "structure_width": abs(qml_level.price - last_lh.price)
                                    },
                                    description=f"Bullish CHOCH: {current.swing_type.value} @ {current.price:.2f} broke downtrend resistance @ {last_lh.price:.2f}"
                                ))
                                choch_detected = True

            # --- Enhanced BOS Detection (Only if no CHOCH detected) ---
            if not choch_detected and current.swing_type == SwingType.HH:
                # Look for previous HH to break
                prev_highs = self._find_all_swings(structure, SwingType.HH, i, limit=2)
                for prev_hh in prev_highs:
                    if current.price > prev_hh.price:
                        # Find intermediate low between the highs
                        intermediate_low = None
                        for j in range(i - 1, -1, -1):
                            if (structure[j].swing_type == SwingType.HL and 
                                structure[j].timestamp > prev_hh.timestamp):
                                intermediate_low = structure[j]
                                break
                        
                        if intermediate_low and self._validate_bos_pattern(current, prev_hh, intermediate_low):
                            confidence, quality_score = self._calculate_confidence(current, prev_hh, intermediate_low, EventType.BOS)
                            if confidence >= self.confidence_threshold:
                                events.append(MarketEvent(
                                    event_type=EventType.BOS,
                                    direction="Bullish",
                                    timestamp=current.timestamp,
                                    price=current.price,
                                    confidence=confidence,
                                    broken_level={"name": "TJL1", "timestamp": prev_hh.timestamp, "price": prev_hh.price},
                                    context={
                                        "a_plus_entry": {"name": "TJL2", "timestamp": intermediate_low.timestamp, "price": intermediate_low.price},
                                        "quality_score": quality_score,
                                        "structure_width": abs(intermediate_low.price - prev_hh.price)
                                    },
                                    description=f"Bullish BOS: HH @ {current.price:.2f} broke previous HH @ {prev_hh.price:.2f}"
                                ))
                                break  # Only take the first valid BOS

            elif not choch_detected and current.swing_type == SwingType.LL:
                # Look for previous LL to break
                prev_lows = self._find_all_swings(structure, SwingType.LL, i, limit=2)
                for prev_ll in prev_lows:
                    if current.price < prev_ll.price:
                        # Find intermediate high between the lows
                        intermediate_high = None
                        for j in range(i - 1, -1, -1):
                            if (structure[j].swing_type == SwingType.LH and 
                                structure[j].timestamp > prev_ll.timestamp):
                                intermediate_high = structure[j]
                                break
                        
                        if intermediate_high and self._validate_bos_pattern(current, prev_ll, intermediate_high):
                            confidence, quality_score = self._calculate_confidence(current, prev_ll, intermediate_high, EventType.BOS)
                            if confidence >= self.confidence_threshold:
                                events.append(MarketEvent(
                                    event_type=EventType.BOS,
                                    direction="Bearish",
                                    timestamp=current.timestamp,
                                    price=current.price,
                                    confidence=confidence,
                                    broken_level={"name": "TJL1", "timestamp": prev_ll.timestamp, "price": prev_ll.price},
                                    context={
                                        "a_plus_entry": {"name": "TJL2", "timestamp": intermediate_high.timestamp, "price": intermediate_high.price},
                                        "quality_score": quality_score,
                                        "structure_width": abs(intermediate_high.price - prev_ll.price)
                                    },
                                    description=f"Bearish BOS: LL @ {current.price:.2f} broke previous LL @ {prev_ll.price:.2f}"
                                ))
                                break  # Only take the first valid BOS

        # Remove duplicate events that might occur at similar times/prices
        filtered_events = []
        for event in events:
            is_duplicate = False
            for existing in filtered_events:
                if (abs((event.timestamp - existing.timestamp).total_seconds()) < 3600 and  # Within 1 hour
                    abs(event.price - existing.price) < 5 and  # Within 5 points
                    event.event_type == existing.event_type and
                    event.direction == existing.direction):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_events.append(event)

        return filtered_events

    def get_high_quality_events(self, structure_data: List[Dict], require_retracement: bool = True, 
                               require_reversal: bool = True) -> List[MarketEvent]:
        """
        Get high-quality market events that require retracement and reversal confirmation.
        This prevents immediate entries on structure breaks and ensures proper setups.
        """
        # First get all events
        all_events = self.get_market_events(structure_data)
        
        if not all_events:
            return []
        
        # Convert structure data to StructurePoint objects for analysis
        structure = [StructurePoint(pd.Timestamp(p["timestamp"]), p["price"], SwingType(p["type"])) 
                    for p in structure_data]
        
        high_quality_events = []
        
        for event in all_events:
            # Find the index of this event in the structure
            event_index = None
            for i, point in enumerate(structure):
                if (point.timestamp == event.timestamp and 
                    abs(point.price - event.price) < 0.001):
                    event_index = i
                    break
            
            if event_index is None:
                continue  # Skip if we can't find the event in structure
            
            # Check retracement confirmation
            retracement_confirmed = True
            if require_retracement:
                retracement_confirmed = self._check_retracement_confirmation(event, structure, event_index)
            
            # Check reversal confirmation
            reversal_confirmed = True
            if require_reversal:
                reversal_confirmed = self._check_reversal_confirmation(event, structure, event_index)
            
            # Only include events with proper confirmation
            if retracement_confirmed and reversal_confirmed:
                # Add confirmation flags to the event context
                event.context["retracement_confirmed"] = retracement_confirmed
                event.context["reversal_confirmed"] = reversal_confirmed
                event.context["quality_score"] = "A+"
                high_quality_events.append(event)
        
        return high_quality_events

    def debug_analysis(self, structure_data: List[Dict], focus_index: int = None) -> None:
        """Debug method to understand what's happening at specific points"""
        if len(structure_data) < 4:
            print("Not enough data for analysis")
            return
        
        structure = [StructurePoint(pd.Timestamp(p["timestamp"]), p["price"], SwingType(p["type"])) 
                    for p in structure_data]
        
        print(f"\n=== DEBUG ANALYSIS ===")
        print(f"Total structure points: {len(structure)}")
        
        if focus_index is None:
            # Show trend states for all points
            for i in range(2, min(10, len(structure))):  # First 10 points
                current = structure[i]
                trend = self._get_trend_state(structure, i)
                print(f"Index {i}: {current.swing_type.value} @ {current.price:.2f} on {current.timestamp} - Trend: {trend}")
        else:
            # Focus on specific index
            if focus_index < len(structure):
                current = structure[focus_index]
                trend = self._get_trend_state(structure, focus_index)
                print(f"\nFOCUS - Index {focus_index}: {current.swing_type.value} @ {current.price:.2f} - Trend: {trend}")
                
                # Show recent history
                lookback = min(6, focus_index)
                recent = structure[max(0, focus_index - lookback):focus_index]
                print(f"Recent history ({len(recent)} points):")
                for j, point in enumerate(recent):
                    print(f"  {focus_index - len(recent) + j}: {point.swing_type.value} @ {point.price:.2f} on {point.timestamp}")
                
                # Check for potential CHOCH
                if trend == "uptrend":
                    last_hl = self._find_last_swing(structure, SwingType.HL, focus_index)
                    if last_hl:
                        print(f"Last HL: {last_hl.price:.2f} on {last_hl.timestamp}")
                        if current.price < last_hl.price:
                            print(f"*** SHOULD BE CHOCH: {current.swing_type.value} @ {current.price:.2f} broke HL @ {last_hl.price:.2f}")
        
        print("=== END DEBUG ===\n")

    def get_event_statistics(self, events: List[MarketEvent]) -> Dict:
        """Get statistics about detected events"""
        if not events:
            return {"total": 0}
        
        stats = {
            "total": len(events),
            "bos_count": len([e for e in events if e.event_type == EventType.BOS]),
            "choch_count": len([e for e in events if e.event_type == EventType.CHOCH]),
            "bullish_count": len([e for e in events if e.direction == "Bullish"]),
            "bearish_count": len([e for e in events if e.direction == "Bearish"]),
            "avg_confidence": sum(e.confidence for e in events) / len(events),
            "high_confidence_count": len([e for e in events if e.confidence > 0.7])
        }
        
        return stats