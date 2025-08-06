import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class PatternType(Enum):
    BULLISH_ENGULFING = "Bullish Engulfing"
    BEARISH_ENGULFING = "Bearish Engulfing"
    HAMMER = "Hammer"
    HANGING_MAN = "Hanging Man"
    INVERTED_HAMMER = "Inverted Hammer"
    SHOOTING_STAR = "Shooting Star"
    DOJI = "Doji"
    SPINNING_TOP = "Spinning Top"
    MORNING_STAR = "Morning Star"
    EVENING_STAR = "Evening Star"
    THREE_WHITE_SOLDIERS = "Three White Soldiers"
    THREE_BLACK_CROWS = "Three Black Crows"
    TWEEZER_TOP = "Tweezer Top"
    TWEEZER_BOTTOM = "Tweezer Bottom"

@dataclass
class CandlestickPattern:
    pattern_type: PatternType
    timestamp: pd.Timestamp
    price: float
    direction: str  # 'Bullish' or 'Bearish'
    confidence: float
    candle_indices: List[int]  # Indices of candles involved in pattern
    description: str

class BadDataError(Exception):
    """Custom exception for bad data"""
    pass

class CandlestickPatternDetector:
    """
    Advanced candlestick pattern detection with configurable parameters
    """
    
    def __init__(self):
        self.patterns = []
        
    def is_bearish_candle(self, candle: pd.Series) -> bool:
        """Check if candle is bearish (close < open)"""
        return candle["Close"] < candle["Open"]

    def is_bullish_candle(self, candle: pd.Series) -> bool:
        """Check if candle is bullish (close > open)"""
        return candle["Close"] > candle["Open"]
    
    def is_doji_candle(self, candle: pd.Series, body_threshold: float = 0.02) -> bool:
        """Check if candle is a doji (very small body)"""
        candle_length = candle["High"] - candle["Low"]
        if candle_length == 0:
            return False
        body_size = abs(candle["Close"] - candle["Open"])
        return (body_size / candle_length) <= body_threshold

    def get_candle_body_size(self, candle: pd.Series) -> float:
        """Get the body size as percentage of total candle range"""
        candle_length = candle["High"] - candle["Low"]
        if candle_length == 0:
            return 0
        body_size = abs(candle["Close"] - candle["Open"])
        return body_size / candle_length
    
    def get_upper_wick_size(self, candle: pd.Series) -> float:
        """Get upper wick size as percentage of total candle range"""
        candle_length = candle["High"] - candle["Low"]
        if candle_length == 0:
            return 0
        if self.is_bullish_candle(candle):
            upper_wick = candle["High"] - candle["Close"]
        else:
            upper_wick = candle["High"] - candle["Open"]
        return upper_wick / candle_length
    
    def get_lower_wick_size(self, candle: pd.Series) -> float:
        """Get lower wick size as percentage of total candle range"""
        candle_length = candle["High"] - candle["Low"]
        if candle_length == 0:
            return 0
        if self.is_bullish_candle(candle):
            lower_wick = candle["Open"] - candle["Low"]
        else:
            lower_wick = candle["Close"] - candle["Low"]
        return lower_wick / candle_length

    def is_bullish_engulfing(self, candles: pd.DataFrame, pos: int = -1) -> bool:
        """Detect bullish engulfing pattern"""
        if len(candles) < 2:
            raise BadDataError("Minimum two candles required")
        
        curr_candle = candles.iloc[pos]
        prev_candle = candles.iloc[pos-1]

        # Check for pattern
        if (self.is_bearish_candle(prev_candle) and 
            self.is_bullish_candle(curr_candle) and
            curr_candle["Close"] > prev_candle["Open"] and 
            curr_candle["Open"] < prev_candle["Close"]):
            return True
        return False
    
    def is_bearish_engulfing(self, candles: pd.DataFrame, pos: int = -1) -> bool:
        """Detect bearish engulfing pattern"""
        if len(candles) < 2:
            raise BadDataError("Minimum two candles required")
        
        curr_candle = candles.iloc[pos]
        prev_candle = candles.iloc[pos-1]

        # Check for pattern
        if (self.is_bullish_candle(prev_candle) and 
            self.is_bearish_candle(curr_candle) and
            curr_candle["Close"] < prev_candle["Open"] and 
            curr_candle["Open"] > prev_candle["Close"]):
            return True
        return False
    
    def is_hammer_candle(self, candles: pd.DataFrame, pos: int = -1, 
                        lower_wick: float = 0.6, body: float = 0.2, upper_wick: float = 0.2) -> bool:
        """Detect hammer candle pattern"""
        if len(candles) < 1:
            raise BadDataError("Minimum one candle required")
        
        curr_candle = candles.iloc[pos]
        candle_length = curr_candle["High"] - curr_candle["Low"]
        
        if candle_length == 0:
            return False
        
        if self.is_bullish_candle(curr_candle):
            candle_upper_wick = curr_candle["High"] - curr_candle["Close"]
            candle_lower_wick = curr_candle["Open"] - curr_candle["Low"]
            candle_body = curr_candle["Close"] - curr_candle["Open"]
        else:
            candle_upper_wick = curr_candle["High"] - curr_candle["Open"]
            candle_lower_wick = curr_candle["Close"] - curr_candle["Low"]
            candle_body = curr_candle["Open"] - curr_candle["Close"]
        
        # Check hammer criteria
        if (candle_body <= body * candle_length and 
            candle_upper_wick <= upper_wick * candle_length and
            candle_lower_wick >= lower_wick * candle_length):
            return True
        return False
    
    def is_inverted_hammer_candle(self, candles: pd.DataFrame, pos: int = -1,
                                 lower_wick: float = 0.2, body: float = 0.2, upper_wick: float = 0.6) -> bool:
        """Detect inverted hammer candle pattern"""
        if len(candles) < 1:
            raise BadDataError("Minimum one candle required")
        
        curr_candle = candles.iloc[pos]
        candle_length = curr_candle["High"] - curr_candle["Low"]
        
        if candle_length == 0:
            return False
        
        if self.is_bullish_candle(curr_candle):
            candle_body = curr_candle["Close"] - curr_candle["Open"]
            candle_upper_wick = curr_candle["High"] - curr_candle["Close"]
            candle_lower_wick = curr_candle["Open"] - curr_candle["Low"]
        else:
            candle_body = curr_candle["Open"] - curr_candle["Close"]
            candle_upper_wick = curr_candle["High"] - curr_candle["Open"]
            candle_lower_wick = curr_candle["Close"] - curr_candle["Low"]
        
        # Check inverted hammer criteria
        if (candle_body <= body * candle_length and 
            candle_lower_wick <= lower_wick * candle_length and
            candle_upper_wick >= upper_wick * candle_length):
            return True
        return False
    
    def is_hanging_man_candle(self, candles: pd.DataFrame, pos: int = -1,
                             lower_wick: float = 0.6, body: float = 0.2, upper_wick: float = 0.2) -> bool:
        """Detect hanging man candle pattern (same as hammer but bearish context)"""
        # Hanging man has same structure as hammer but appears in uptrend
        return self.is_hammer_candle(candles, pos, lower_wick, body, upper_wick)
    
    def is_shooting_star_candle(self, candles: pd.DataFrame, pos: int = -1,
                               lower_wick: float = 0.2, body: float = 0.2, upper_wick: float = 0.6) -> bool:
        """Detect shooting star candle pattern (same as inverted hammer but bearish context)"""
        # Shooting star has same structure as inverted hammer but appears in uptrend
        return self.is_inverted_hammer_candle(candles, pos, lower_wick, body, upper_wick)
    
    def is_doji_pattern(self, candles: pd.DataFrame, pos: int = -1,
                       lower_wick: float = 0.4, body: float = 0.02, upper_wick: float = 0.4) -> bool:
        """Detect doji candle pattern"""
        if len(candles) < 1:
            raise BadDataError("Minimum one candle required")
        
        curr_candle = candles.iloc[pos]
        candle_length = curr_candle["High"] - curr_candle["Low"]
        
        if candle_length == 0:
            return False
        
        candle_body = abs(curr_candle["Close"] - curr_candle["Open"])
        candle_upper_wick = curr_candle["High"] - max(curr_candle["Open"], curr_candle["Close"])
        candle_lower_wick = min(curr_candle["Open"], curr_candle["Close"]) - curr_candle["Low"]
        
        # Check doji criteria
        if (candle_body <= body * candle_length and 
            candle_upper_wick >= upper_wick * candle_length and
            candle_lower_wick >= lower_wick * candle_length):
            return True
        return False
    
    def is_spinning_top(self, candles: pd.DataFrame, pos: int = -1,
                       lower_wick: float = 0.3, body: float = 0.1, upper_wick: float = 0.3) -> bool:
        """Detect spinning top pattern - small body with upper and lower wicks"""
        if len(candles) < 1:
            raise BadDataError("Minimum one candle required")
        
        curr_candle = candles.iloc[pos]
        candle_length = curr_candle["High"] - curr_candle["Low"]
        
        if candle_length == 0:
            return False
        
        candle_body = abs(curr_candle["Close"] - curr_candle["Open"])
        candle_upper_wick = curr_candle["High"] - max(curr_candle["Open"], curr_candle["Close"])
        candle_lower_wick = min(curr_candle["Open"], curr_candle["Close"]) - curr_candle["Low"]
        
        # Check spinning top criteria - small body with both upper and lower wicks
        if (candle_body <= body * candle_length and
            candle_upper_wick >= upper_wick * candle_length and
            candle_lower_wick >= lower_wick * candle_length):
            return True
        return False

    def is_morning_star(self, candles: pd.DataFrame, pos: int = -1) -> bool:
        """Detect morning star pattern (3-candle bullish reversal)"""
        if len(candles) < 3:
            raise BadDataError("Minimum three candles required")
        
        candle1 = candles.iloc[pos-2]  # First candle (bearish)
        candle2 = candles.iloc[pos-1]  # Middle candle (small body)
        candle3 = candles.iloc[pos]    # Last candle (bullish)
        
        # Check for pattern
        if (self.is_bearish_candle(candle1) and
            self.get_candle_body_size(candle2) < 0.3 and  # Small body
            self.is_bullish_candle(candle3) and
            candle3["Close"] > (candle1["Open"] + candle1["Close"]) / 2):  # Close above midpoint of first candle
            return True
        return False
    
    def is_evening_star(self, candles: pd.DataFrame, pos: int = -1) -> bool:
        """Detect evening star pattern (3-candle bearish reversal)"""
        if len(candles) < 3:
            raise BadDataError("Minimum three candles required")
        
        candle1 = candles.iloc[pos-2]  # First candle (bullish)
        candle2 = candles.iloc[pos-1]  # Middle candle (small body)
        candle3 = candles.iloc[pos]    # Last candle (bearish)
        
        # Check for pattern
        if (self.is_bullish_candle(candle1) and
            self.get_candle_body_size(candle2) < 0.3 and  # Small body
            self.is_bearish_candle(candle3) and
            candle3["Close"] < (candle1["Open"] + candle1["Close"]) / 2):  # Close below midpoint of first candle
            return True
        return False
    
    def is_three_white_soldiers(self, candles: pd.DataFrame, pos: int = -1) -> bool:
        """Detect three white soldiers pattern (3 consecutive bullish candles)"""
        if len(candles) < 3:
            raise BadDataError("Minimum three candles required")
        
        candle1 = candles.iloc[pos-2]
        candle2 = candles.iloc[pos-1]
        candle3 = candles.iloc[pos]
        
        # Check for three consecutive bullish candles with increasing closes
        if (self.is_bullish_candle(candle1) and
            self.is_bullish_candle(candle2) and
            self.is_bullish_candle(candle3) and
            candle2["Close"] > candle1["Close"] and
            candle3["Close"] > candle2["Close"] and
            candle2["Open"] > candle1["Open"] and
            candle3["Open"] > candle2["Open"]):
            return True
        return False
    
    def is_three_black_crows(self, candles: pd.DataFrame, pos: int = -1) -> bool:
        """Detect three black crows pattern (3 consecutive bearish candles)"""
        if len(candles) < 3:
            raise BadDataError("Minimum three candles required")
        
        candle1 = candles.iloc[pos-2]
        candle2 = candles.iloc[pos-1]
        candle3 = candles.iloc[pos]
        
        # Check for three consecutive bearish candles with decreasing closes
        if (self.is_bearish_candle(candle1) and
            self.is_bearish_candle(candle2) and
            self.is_bearish_candle(candle3) and
            candle2["Close"] < candle1["Close"] and
            candle3["Close"] < candle2["Close"] and
            candle2["Open"] < candle1["Open"] and
            candle3["Open"] < candle2["Open"]):
            return True
        return False
    
    def is_tweezer_top(self, candles: pd.DataFrame, pos: int = -1) -> bool:
        """Detect tweezer top pattern (two candles with similar highs)"""
        if len(candles) < 2:
            raise BadDataError("Minimum two candles required")
        
        candle1 = candles.iloc[pos-1]
        candle2 = candles.iloc[pos]
        
        # Check for similar highs with tolerance
        high_tolerance = abs(candle1["High"] - candle2["High"]) / candle1["High"]
        
        # Both candles should have significant upper wicks and similar highs
        if (high_tolerance <= 0.002 and  # Within 0.2% tolerance
            self.is_bullish_candle(candle1) and
            self.is_bearish_candle(candle2)):
            return True
        return False
    
    def is_tweezer_bottom(self, candles: pd.DataFrame, pos: int = -1) -> bool:
        """Detect tweezer bottom pattern (two candles with similar lows)"""
        if len(candles) < 2:
            raise BadDataError("Minimum two candles required")
        
        candle1 = candles.iloc[pos-1]
        candle2 = candles.iloc[pos]
        
        # Check for similar lows with tolerance
        low_tolerance = abs(candle1["Low"] - candle2["Low"]) / candle1["Low"]
        
        # Both candles should have significant lower wicks and similar lows
        if (low_tolerance <= 0.002 and  # Within 0.2% tolerance
            self.is_bearish_candle(candle1) and
            self.is_bullish_candle(candle2)):
            return True
        return False

    def _get_trend_context(self, df: pd.DataFrame, pos: int, lookback: int = 5) -> str:
        """Determine trend context for pattern classification"""
        if pos < lookback:
            return "neutral"
        
        # Look at price movement over lookback period
        current_price = df.iloc[pos]['Close']
        past_price = df.iloc[pos - lookback]['Close']
        
        price_change = (current_price - past_price) / past_price
        
        if price_change > 0.02:  # 2% increase
            return "uptrend"
        elif price_change < -0.02:  # 2% decrease
            return "downtrend"
        else:
            return "neutral"
    
    def _detect_timeframe(self, df: pd.DataFrame) -> str:
        """Detect timeframe from DataFrame index"""
        if len(df) < 2:
            return "unknown"
        
        # Calculate time difference between consecutive candles
        time_diff = df.index[1] - df.index[0]
        minutes = time_diff.total_seconds() / 60
        
        if minutes <= 1:
            return "1M"
        elif minutes <= 5:
            return "5M"
        elif minutes <= 15:
            return "15M"
        elif minutes <= 30:
            return "30M"
        elif minutes <= 60:
            return "1H"
        elif minutes <= 240:
            return "4H"
        elif minutes <= 1440:
            return "1D"
        else:
            return "unknown"
    
    def _get_timeframe_confidence_multiplier(self, pattern_type: PatternType, timeframe: str) -> float:
        """Get confidence multiplier based on pattern type and timeframe"""
        # Best timeframes for each pattern
        optimal_timeframes = {
            PatternType.HAMMER: ["15M", "1H"],
            PatternType.SHOOTING_STAR: ["15M", "1H"],
            PatternType.BULLISH_ENGULFING: ["15M", "1H", "4H"],
            PatternType.BEARISH_ENGULFING: ["15M", "1H", "4H"],
            PatternType.DOJI: ["15M"],
            PatternType.SPINNING_TOP: ["15M"],
            PatternType.MORNING_STAR: ["1H", "4H"],
            PatternType.EVENING_STAR: ["1H", "4H"],
            PatternType.THREE_WHITE_SOLDIERS: ["1H", "4H", "1D"],
            PatternType.THREE_BLACK_CROWS: ["1H", "4H", "1D"],
            PatternType.TWEEZER_TOP: ["15M", "1H"],
            PatternType.TWEEZER_BOTTOM: ["15M", "1H"]
        }
        
        if pattern_type in optimal_timeframes:
            if timeframe in optimal_timeframes[pattern_type]:
                return 1.2  # 20% boost for optimal timeframes
            else:
                return 0.9  # 10% reduction for non-optimal timeframes
        
        return 1.0  # No change for unknown patterns

    def detect_patterns(self, df: pd.DataFrame, min_confidence: float = 0.6) -> List[CandlestickPattern]:
        """
        Detect all candlestick patterns in the given dataframe with timeframe-aware confidence
        """
        patterns = []
        timeframe = self._detect_timeframe(df)
        
        for i in range(2, len(df)):  # Start from index 2 to allow for 3-candle patterns
            candles_subset = df.iloc[:i+1]
            current_time = df.index[i]
            current_price = df.iloc[i]['Close']
            trend_context = self._get_trend_context(df, i)
            
            # Single candle patterns with trend context
            
            # Hammer (bullish reversal in downtrend)
            if self.is_hammer_candle(candles_subset, pos=-1):
                confidence = self._calculate_hammer_confidence(candles_subset, -1)
                # Apply timeframe multiplier
                confidence *= self._get_timeframe_confidence_multiplier(PatternType.HAMMER, timeframe)
                confidence = min(0.95, confidence)  # Cap at 95%
                if confidence >= min_confidence:
                    patterns.append(CandlestickPattern(
                        pattern_type=PatternType.HAMMER,
                        timestamp=current_time,
                        price=current_price,
                        direction='Bullish',
                        confidence=confidence,
                        candle_indices=[i],
                        description="Hammer pattern - potential bullish reversal"
                    ))
            
            # Hanging Man (bearish reversal in uptrend)
            if self.is_hanging_man_candle(candles_subset, pos=-1) and trend_context == "uptrend":
                confidence = self._calculate_hammer_confidence(candles_subset, -1)  # Same calculation as hammer
                # Apply timeframe multiplier
                confidence *= self._get_timeframe_confidence_multiplier(PatternType.HANGING_MAN, timeframe)
                confidence = min(0.95, confidence)  # Cap at 95%
                if confidence >= min_confidence:
                    patterns.append(CandlestickPattern(
                        pattern_type=PatternType.HANGING_MAN,
                        timestamp=current_time,
                        price=current_price,
                        direction='Bearish',
                        confidence=confidence,
                        candle_indices=[i],
                        description="Hanging man pattern - potential bearish reversal"
                    ))
            
            # Inverted Hammer (bullish reversal in downtrend)
            if self.is_inverted_hammer_candle(candles_subset, pos=-1):
                confidence = self._calculate_inverted_hammer_confidence(candles_subset, -1)
                # Apply timeframe multiplier
                confidence *= self._get_timeframe_confidence_multiplier(PatternType.INVERTED_HAMMER, timeframe)
                confidence = min(0.95, confidence)  # Cap at 95%
                if confidence >= min_confidence:
                    patterns.append(CandlestickPattern(
                        pattern_type=PatternType.INVERTED_HAMMER,
                        timestamp=current_time,
                        price=current_price,
                        direction='Bullish',
                        confidence=confidence,
                        candle_indices=[i],
                        description="Inverted hammer pattern - potential bullish reversal"
                    ))
            
            # Shooting Star (bearish reversal in uptrend)
            if self.is_shooting_star_candle(candles_subset, pos=-1) and trend_context == "uptrend":
                confidence = self._calculate_inverted_hammer_confidence(candles_subset, -1)  # Same calculation as inverted hammer
                # Apply timeframe multiplier
                confidence *= self._get_timeframe_confidence_multiplier(PatternType.SHOOTING_STAR, timeframe)
                confidence = min(0.95, confidence)  # Cap at 95%
                if confidence >= min_confidence:
                    patterns.append(CandlestickPattern(
                        pattern_type=PatternType.SHOOTING_STAR,
                        timestamp=current_time,
                        price=current_price,
                        direction='Bearish',
                        confidence=confidence,
                        candle_indices=[i],
                        description="Shooting star pattern - potential bearish reversal"
                    ))
            
            # Doji (indecision, potential reversal)
            if self.is_doji_pattern(candles_subset, pos=-1):
                confidence = self._calculate_doji_confidence(candles_subset, -1)
                # Apply timeframe multiplier
                confidence *= self._get_timeframe_confidence_multiplier(PatternType.DOJI, timeframe)
                confidence = min(0.95, confidence)  # Cap at 95%
                if confidence >= min_confidence:
                    patterns.append(CandlestickPattern(
                        pattern_type=PatternType.DOJI,
                        timestamp=current_time,
                        price=current_price,
                        direction='Neutral',
                        confidence=confidence,
                        candle_indices=[i],
                        description="Doji pattern - indecision, potential reversal"
                    ))
            
            # Spinning Top (indecision with confirmation needed)
            if self.is_spinning_top(candles_subset, pos=-1):
                confidence = self._calculate_doji_confidence(candles_subset, -1)  # Similar to doji
                # Apply timeframe multiplier
                confidence *= self._get_timeframe_confidence_multiplier(PatternType.SPINNING_TOP, timeframe)
                confidence = min(0.95, confidence)  # Cap at 95%
                if confidence >= min_confidence:
                    patterns.append(CandlestickPattern(
                        pattern_type=PatternType.SPINNING_TOP,
                        timestamp=current_time,
                        price=current_price,
                        direction='Neutral',
                        confidence=confidence,
                        candle_indices=[i],
                        description="Spinning top pattern - indecision, needs confirmation"
                    ))
            
            # Two candle patterns
            
            # Bullish Engulfing
            if self.is_bullish_engulfing(candles_subset, pos=-1):
                confidence = self._calculate_engulfing_confidence(candles_subset, -1, True)
                # Apply timeframe multiplier
                confidence *= self._get_timeframe_confidence_multiplier(PatternType.BULLISH_ENGULFING, timeframe)
                confidence = min(0.95, confidence)  # Cap at 95%
                if confidence >= min_confidence:
                    patterns.append(CandlestickPattern(
                        pattern_type=PatternType.BULLISH_ENGULFING,
                        timestamp=current_time,
                        price=current_price,
                        direction='Bullish',
                        confidence=confidence,
                        candle_indices=[i-1, i],
                        description="Bullish engulfing pattern - strong bullish reversal"
                    ))
            
            # Bearish Engulfing
            if self.is_bearish_engulfing(candles_subset, pos=-1):
                confidence = self._calculate_engulfing_confidence(candles_subset, -1, False)
                # Apply timeframe multiplier
                confidence *= self._get_timeframe_confidence_multiplier(PatternType.BEARISH_ENGULFING, timeframe)
                confidence = min(0.95, confidence)  # Cap at 95%
                if confidence >= min_confidence:
                    patterns.append(CandlestickPattern(
                        pattern_type=PatternType.BEARISH_ENGULFING,
                        timestamp=current_time,
                        price=current_price,
                        direction='Bearish',
                        confidence=confidence,
                        candle_indices=[i-1, i],
                        description="Bearish engulfing pattern - strong bearish reversal"
                    ))
            
            # Three candle patterns
            if i >= 2:  # Need at least 3 candles
                
                # Morning Star
                if self.is_morning_star(candles_subset, pos=-1):
                    confidence = self._calculate_star_confidence(candles_subset, -1, True)
                    # Apply timeframe multiplier
                    confidence *= self._get_timeframe_confidence_multiplier(PatternType.MORNING_STAR, timeframe)
                    confidence = min(0.95, confidence)  # Cap at 95%
                    if confidence >= min_confidence:
                        patterns.append(CandlestickPattern(
                            pattern_type=PatternType.MORNING_STAR,
                            timestamp=current_time,
                            price=current_price,
                            direction='Bullish',
                            confidence=confidence,
                            candle_indices=[i-2, i-1, i],
                            description="Morning star pattern - strong bullish reversal"
                        ))
                
                # Evening Star
                if self.is_evening_star(candles_subset, pos=-1):
                    confidence = self._calculate_star_confidence(candles_subset, -1, False)
                    # Apply timeframe multiplier
                    confidence *= self._get_timeframe_confidence_multiplier(PatternType.EVENING_STAR, timeframe)
                    confidence = min(0.95, confidence)  # Cap at 95%
                    if confidence >= min_confidence:
                        patterns.append(CandlestickPattern(
                            pattern_type=PatternType.EVENING_STAR,
                            timestamp=current_time,
                            price=current_price,
                            direction='Bearish',
                            confidence=confidence,
                            candle_indices=[i-2, i-1, i],
                            description="Evening star pattern - strong bearish reversal"
                        ))
                
                # Three White Soldiers (trend confirmation)
                if self.is_three_white_soldiers(candles_subset, pos=-1):
                    confidence = self._calculate_soldiers_confidence(candles_subset, -1, True)
                    # Apply timeframe multiplier
                    confidence *= self._get_timeframe_confidence_multiplier(PatternType.THREE_WHITE_SOLDIERS, timeframe)
                    confidence = min(0.95, confidence)  # Cap at 95%
                    if confidence >= min_confidence:
                        patterns.append(CandlestickPattern(
                            pattern_type=PatternType.THREE_WHITE_SOLDIERS,
                            timestamp=current_time,
                            price=current_price,
                            direction='Bullish',
                            confidence=confidence,
                            candle_indices=[i-2, i-1, i],
                            description="Three white soldiers pattern - strong bullish trend confirmation"
                        ))
                
                # Three Black Crows (trend confirmation)
                if self.is_three_black_crows(candles_subset, pos=-1):
                    confidence = self._calculate_soldiers_confidence(candles_subset, -1, False)
                    # Apply timeframe multiplier
                    confidence *= self._get_timeframe_confidence_multiplier(PatternType.THREE_BLACK_CROWS, timeframe)
                    confidence = min(0.95, confidence)  # Cap at 95%
                    if confidence >= min_confidence:
                        patterns.append(CandlestickPattern(
                            pattern_type=PatternType.THREE_BLACK_CROWS,
                            timestamp=current_time,
                            price=current_price,
                            direction='Bearish',
                            confidence=confidence,
                            candle_indices=[i-2, i-1, i],
                            description="Three black crows pattern - strong bearish trend confirmation"
                        ))
            
            # Two candle patterns (additional)
            if i >= 1:  # Need at least 2 candles
                
                # Tweezer Top (strong rejection)
                if self.is_tweezer_top(candles_subset, pos=-1):
                    confidence = self._calculate_tweezer_confidence(candles_subset, -1, True)
                    # Apply timeframe multiplier
                    confidence *= self._get_timeframe_confidence_multiplier(PatternType.TWEEZER_TOP, timeframe)
                    confidence = min(0.95, confidence)  # Cap at 95%
                    if confidence >= min_confidence:
                        patterns.append(CandlestickPattern(
                            pattern_type=PatternType.TWEEZER_TOP,
                            timestamp=current_time,
                            price=current_price,
                            direction='Bearish',
                            confidence=confidence,
                            candle_indices=[i-1, i],
                            description="Tweezer top pattern - strong resistance rejection"
                        ))
                
                # Tweezer Bottom (strong rejection)
                if self.is_tweezer_bottom(candles_subset, pos=-1):
                    confidence = self._calculate_tweezer_confidence(candles_subset, -1, False)
                    # Apply timeframe multiplier
                    confidence *= self._get_timeframe_confidence_multiplier(PatternType.TWEEZER_BOTTOM, timeframe)
                    confidence = min(0.95, confidence)  # Cap at 95%
                    if confidence >= min_confidence:
                        patterns.append(CandlestickPattern(
                            pattern_type=PatternType.TWEEZER_BOTTOM,
                            timestamp=current_time,
                            price=current_price,
                            direction='Bullish',
                            confidence=confidence,
                            candle_indices=[i-1, i],
                            description="Tweezer bottom pattern - strong support rejection"
                        ))
        
        return patterns

    def _calculate_hammer_confidence(self, candles: pd.DataFrame, pos: int) -> float:
        """Calculate confidence score for hammer pattern"""
        candle = candles.iloc[pos]
        lower_wick_ratio = self.get_lower_wick_size(candle)
        upper_wick_ratio = self.get_upper_wick_size(candle)
        body_ratio = self.get_candle_body_size(candle)
        
        # Higher confidence for longer lower wick and smaller upper wick
        confidence = min(0.95, lower_wick_ratio * 1.5 + (1 - upper_wick_ratio) * 0.3 + (1 - body_ratio) * 0.2)
        return max(0.5, confidence)
    
    def _calculate_inverted_hammer_confidence(self, candles: pd.DataFrame, pos: int) -> float:
        """Calculate confidence score for inverted hammer pattern"""
        candle = candles.iloc[pos]
        upper_wick_ratio = self.get_upper_wick_size(candle)
        lower_wick_ratio = self.get_lower_wick_size(candle)
        body_ratio = self.get_candle_body_size(candle)
        
        # Higher confidence for longer upper wick and smaller lower wick
        confidence = min(0.95, upper_wick_ratio * 1.5 + (1 - lower_wick_ratio) * 0.3 + (1 - body_ratio) * 0.2)
        return max(0.5, confidence)
    
    def _calculate_doji_confidence(self, candles: pd.DataFrame, pos: int) -> float:
        """Calculate confidence score for doji pattern"""
        candle = candles.iloc[pos]
        body_ratio = self.get_candle_body_size(candle)
        
        # Higher confidence for smaller body
        confidence = min(0.95, (1 - body_ratio * 10) * 0.8 + 0.2)
        return max(0.5, confidence)
    
    def _calculate_engulfing_confidence(self, candles: pd.DataFrame, pos: int, is_bullish: bool) -> float:
        """Calculate confidence score for engulfing patterns"""
        curr_candle = candles.iloc[pos]
        prev_candle = candles.iloc[pos-1]
        
        # Calculate how much the current candle engulfs the previous
        curr_body = abs(curr_candle['Close'] - curr_candle['Open'])
        prev_body = abs(prev_candle['Close'] - prev_candle['Open'])
        
        if prev_body == 0:
            return 0.5
        
        engulfing_ratio = curr_body / prev_body
        confidence = min(0.95, 0.6 + engulfing_ratio * 0.3)
        return max(0.5, confidence)
    

    
    def _calculate_star_confidence(self, candles: pd.DataFrame, pos: int, is_morning: bool) -> float:
        """Calculate confidence score for star patterns"""
        candle1 = candles.iloc[pos-2]
        candle2 = candles.iloc[pos-1]  # Star candle
        candle3 = candles.iloc[pos]
        
        # Star candle should have small body
        star_body_ratio = self.get_candle_body_size(candle2)
        
        # Third candle should have good body size
        third_body_ratio = self.get_candle_body_size(candle3)
        
        confidence = min(0.95, 0.5 + (1 - star_body_ratio) * 0.3 + third_body_ratio * 0.2)
        return max(0.5, confidence)
    
    def _calculate_soldiers_confidence(self, candles: pd.DataFrame, pos: int, is_bullish: bool) -> float:
        """Calculate confidence score for three soldiers/crows patterns"""
        candle1 = candles.iloc[pos-2]
        candle2 = candles.iloc[pos-1]
        candle3 = candles.iloc[pos]
        
        # Calculate body sizes
        body1 = self.get_candle_body_size(candle1)
        body2 = self.get_candle_body_size(candle2)
        body3 = self.get_candle_body_size(candle3)
        
        # All candles should have substantial bodies
        avg_body_size = (body1 + body2 + body3) / 3
        
        # Calculate price progression consistency
        if is_bullish:
            price_progression = (candle3['Close'] - candle1['Open']) / candle1['Open']
        else:
            price_progression = (candle1['Open'] - candle3['Close']) / candle1['Open']
        
        confidence = min(0.95, 0.6 + avg_body_size * 0.2 + min(price_progression * 10, 0.15))
        return max(0.5, confidence)
    
    def _calculate_tweezer_confidence(self, candles: pd.DataFrame, pos: int, is_top: bool) -> float:
        """Calculate confidence score for tweezer patterns"""
        candle1 = candles.iloc[pos-1]
        candle2 = candles.iloc[pos]
        
        if is_top:
            # Check how close the highs are
            high_similarity = 1 - abs(candle1['High'] - candle2['High']) / candle1['High']
            # Check for upper wicks
            wick1 = self.get_upper_wick_size(candle1)
            wick2 = self.get_upper_wick_size(candle2)
        else:
            # Check how close the lows are
            high_similarity = 1 - abs(candle1['Low'] - candle2['Low']) / candle1['Low']
            # Check for lower wicks
            wick1 = self.get_lower_wick_size(candle1)
            wick2 = self.get_lower_wick_size(candle2)
        
        avg_wick_size = (wick1 + wick2) / 2
        confidence = min(0.95, 0.5 + high_similarity * 0.3 + avg_wick_size * 0.15)
        return max(0.5, confidence)