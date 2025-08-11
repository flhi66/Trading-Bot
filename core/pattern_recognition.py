import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings

@dataclass
class PivotPoint:
    """Represents a pivot point in price data"""
    timestamp: pd.Timestamp
    price: float
    pivot_type: str  # 'high' or 'low'
    percentage_change: float
    start_price: float
    end_price: float

@dataclass
class PriceMovement:
    """Represents a price movement between pivot points"""
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    start_price: float
    end_price: float
    movement_type: str  # 'uptrend' or 'downtrend'
    percentage_change: float
    duration: pd.Timedelta

class PivotPointDetector:
    """Detects pivot points in price data using enhanced algorithms and multi-timeframe analysis"""
    
    def __init__(self, threshold_percentage: float = 0.3, min_duration: str = '5min', 
                 use_multi_timeframe: bool = True, confirmation_periods: int = 5):
        """
        Initialize the pivot point detector
        
        Args:
            threshold_percentage: Minimum percentage change to consider as significant
            min_duration: Minimum duration for a movement to be valid
            use_multi_timeframe: Whether to use multi-timeframe analysis
            confirmation_periods: Number of periods to confirm pivot points
        """
        self.threshold_percentage = threshold_percentage / 100.0
        # Convert 'H' to 'h' to avoid deprecation warning
        min_duration = min_duration.replace('H', 'h') if isinstance(min_duration, str) else min_duration
        self.min_duration = pd.Timedelta(min_duration)
        self.use_multi_timeframe = use_multi_timeframe
        self.confirmation_periods = confirmation_periods
    
    def find_pivot_points(self, df: pd.DataFrame, price_col: str = 'close') -> List[PivotPoint]:
        """
        Find pivot points in the price data using enhanced algorithm
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price data
            
        Returns:
            List of PivotPoint objects
        """
        if df.empty:
            return []
        
        # Use enhanced pivot detection with multiple thresholds
        return self._find_enhanced_pivots(df, price_col)
    
    def _find_enhanced_pivots(self, df: pd.DataFrame, price_col: str) -> List[PivotPoint]:
        """Enhanced pivot detection with better thresholds and confirmation"""
        prices = df[price_col].values
        timestamps = df.index.values
        pivot_points = []
        
        # Use multiple thresholds for different market conditions
        # Start with higher thresholds and gradually reduce if not enough pivots found
        thresholds = [0.01, 0.005, 0.002, 0.001]  # 1%, 0.5%, 0.2%, 0.1%
        
        for threshold in thresholds:
            if len(pivot_points) >= 10:  # Stop if we have enough pivots
                break
                
            for i in range(self.confirmation_periods, len(prices) - self.confirmation_periods):
                current_price = prices[i]
                
                # Check for local high with confirmation
                if (current_price > prices[i-self.confirmation_periods:i].max() and 
                    current_price > prices[i+1:i+self.confirmation_periods+1].max()):
                    
                    # Calculate percentage drop from high
                    future_drops = (current_price - prices[i:i+self.confirmation_periods*2]) / current_price
                    past_drops = (current_price - prices[i-self.confirmation_periods*2:i]) / current_price
                    
                    if len(future_drops) > 0 and len(past_drops) > 0:
                        max_drop = max(future_drops.max(), past_drops.max())
                        
                        if max_drop >= threshold:
                            # Additional confirmation: check if this is a significant high
                            if self._is_significant_pivot(prices, i, 'high', threshold):
                                pivot_points.append(PivotPoint(
                                    timestamp=timestamps[i],
                                    price=current_price,
                                    pivot_type='high',
                                    percentage_change=max_drop,
                                    start_price=prices[i-1],
                                    end_price=prices[i+1]
                                ))
                
                # Check for local low with confirmation
                elif (current_price < prices[i-self.confirmation_periods:i].min() and 
                      current_price < prices[i+1:i+self.confirmation_periods+1].min()):
                    
                    # Calculate percentage rise from low
                    future_rises = (prices[i:i+self.confirmation_periods*2] - current_price) / current_price
                    past_rises = (prices[i-self.confirmation_periods*2:i] - current_price) / current_price
                    
                    if len(future_rises) > 0 and len(past_rises) > 0:
                        max_rise = max(future_rises.max(), past_rises.max())
                        
                        if max_rise >= threshold:
                            # Additional confirmation: check if this is a significant low
                            if self._is_significant_pivot(prices, i, 'low', threshold):
                                pivot_points.append(PivotPoint(
                                    timestamp=timestamps[i],
                                    price=current_price,
                                    pivot_type='low',
                                    percentage_change=max_rise,
                                    start_price=prices[i-1],
                                    end_price=prices[i+1]
                                ))
        
        # Sort by timestamp and remove duplicates
        pivot_points.sort(key=lambda x: x.timestamp)
        return self._remove_duplicate_pivots(pivot_points)
    
    def _is_significant_pivot(self, prices: np.ndarray, idx: int, pivot_type: str, threshold: float) -> bool:
        """Check if a pivot point is significant based on surrounding price action"""
        if idx < 10 or idx >= len(prices) - 10:
            return False
        
        current_price = prices[idx]
        
        if pivot_type == 'high':
            # Check if the high is significantly above surrounding prices
            surrounding_prices = prices[idx-10:idx+11]
            if current_price < np.percentile(surrounding_prices, 80):
                return False
            
            # Check if the drop is sustained
            future_prices = prices[idx:idx+10]
            if len(future_prices) > 0 and (current_price - future_prices.min()) / current_price < threshold:
                return False
                
        else:  # low
            # Check if the low is significantly below surrounding prices
            surrounding_prices = prices[idx-10:idx+11]
            if current_price > np.percentile(surrounding_prices, 20):
                return False
            
            # Check if the rise is sustained
            future_prices = prices[idx:idx+10]
            if len(future_prices) > 0 and (future_prices.max() - current_price) / current_price < threshold:
                return False
        
        return True
    
    def _remove_duplicate_pivots(self, pivot_points: List[PivotPoint]) -> List[PivotPoint]:
        """Remove pivot points that are too close to each other"""
        if len(pivot_points) < 2:
            return pivot_points
        
        filtered_pivots = [pivot_points[0]]
        
        for pivot in pivot_points[1:]:
            # Check if this pivot is too close to the last one
            last_pivot = filtered_pivots[-1]
            
            # Handle different timestamp types
            if hasattr(pivot.timestamp, 'total_seconds'):
                time_diff = abs((pivot.timestamp - last_pivot.timestamp).total_seconds() / 3600)  # hours
            else:
                # Handle numpy timestamps
                time_diff = abs((pd.Timestamp(pivot.timestamp) - pd.Timestamp(last_pivot.timestamp)).total_seconds() / 3600)
            
            price_diff = abs(pivot.price - last_pivot.price) / last_pivot.price
            
            # Keep pivot if it's sufficiently different in time or price
            if time_diff >= 2 or price_diff >= 0.002:  # 2 hours or 0.2% price difference
                filtered_pivots.append(pivot)
        
        return filtered_pivots
    
    def find_movements(self, df: pd.DataFrame, price_col: str = 'close') -> Dict[str, pd.DataFrame]:
        """
        Find price movements between pivot points
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price data
            
        Returns:
            Dictionary with 'uptrends' and 'downtrends' DataFrames
        """
        pivot_points = self.find_pivot_points(df, price_col)
        
        if len(pivot_points) < 2:
            return {'uptrends': pd.DataFrame(), 'downtrends': pd.DataFrame()}
        
        movements = []
        
        for i in range(len(pivot_points) - 1):
            current = pivot_points[i]
            next_point = pivot_points[i + 1]
            
            # Calculate movement
            price_change = next_point.price - current.price
            percentage_change = abs(price_change) / current.price
            duration = next_point.timestamp - current.timestamp
            
            if duration >= self.min_duration and percentage_change >= self.threshold_percentage:
                movement_type = 'uptrend' if price_change > 0 else 'downtrend'
                
                movements.append({
                    'start_timestamp': current.timestamp,
                    'end_timestamp': next_point.timestamp,
                    'start_price': current.price,
                    'end_price': next_point.price,
                    'movement_type': movement_type,
                    'percentage_change': percentage_change,
                    'duration': duration
                })
        
        movements_df = pd.DataFrame(movements)
        
        if movements_df.empty:
            return {'uptrends': pd.DataFrame(), 'downtrends': pd.DataFrame()}
        
        # Split into uptrends and downtrends
        uptrends = movements_df[movements_df['movement_type'] == 'uptrend'].copy()
        downtrends = movements_df[movements_df['movement_type'] == 'downtrend'].copy()
        
        return {
            'uptrends': uptrends,
            'downtrends': downtrends
        }

class PiecewiseLinearizer:
    """Converts price movements into piecewise linear segments"""
    
    def __init__(self, timeframe: str = '5min'):
        """
        Initialize the piecewise linearizer
        
        Args:
            timeframe: Timeframe for interpolation (e.g., '5min', '1H')
        """
        # Convert 'H' to 'h' to avoid deprecation warning
        timeframe = timeframe.replace('H', 'h') if isinstance(timeframe, str) else timeframe
        self.timeframe = pd.Timedelta(timeframe)
    
    def linearize_movements(self, movements: Dict[str, pd.DataFrame], 
                           start_date: Union[str, pd.Timestamp],
                           end_date: Optional[Union[str, pd.Timestamp]] = None,
                           normalize: bool = True) -> pd.Series:
        """
        Convert movements to piecewise linear series
        
        Args:
            movements: Dictionary with 'uptrends' and 'downtrends' DataFrames
            start_date: Start date for the series
            end_date: End date for the series (if None, uses start_date + 24H)
            normalize: Whether to normalize the series
            
        Returns:
            Piecewise linear series
        """
        start_ts = pd.Timestamp(start_date)
        if end_date is None:
            end_ts = start_ts + pd.Timedelta('24H')
        else:
            end_ts = pd.Timestamp(end_date)
        
        # Combine all movements
        all_movements = pd.concat([
            movements['uptrends'],
            movements['downtrends']
        ], ignore_index=True).sort_values('start_timestamp')
        
        # Filter movements within the date range
        mask = (all_movements['start_timestamp'] >= start_ts) & (all_movements['end_timestamp'] <= end_ts)
        relevant_movements = all_movements[mask].copy()
        
        if relevant_movements.empty:
            return pd.Series(dtype=float)
        
        # Create time grid
        time_grid = pd.date_range(start=start_ts, end=end_ts, freq=self.timeframe)
        
        # Initialize result series
        result = pd.Series(index=time_grid, dtype=float)
        
        # Fill values using piecewise linear interpolation
        for _, movement in relevant_movements.iterrows():
            start_idx = movement['start_timestamp']
            end_idx = movement['end_timestamp']
            start_price = movement['start_price']
            end_price = movement['end_price']
            
            # Find time grid indices
            start_mask = time_grid >= start_idx
            end_mask = time_grid <= end_idx
            valid_mask = start_mask & end_mask
            
            if not valid_mask.any():
                continue
            
            valid_times = time_grid[valid_mask]
            
            # Linear interpolation
            for t in valid_times:
                if start_idx == end_idx:
                    result[t] = start_price
                else:
                    # Calculate interpolation factor
                    factor = (t - start_idx) / (end_idx - start_idx)
                    result[t] = start_price + factor * (end_price - start_price)
        
        # Forward fill any remaining NaN values
        result = result.ffill()
        
        # Normalize if requested
        if normalize and not result.empty and result.iloc[0] != 0:
            result = result / result.iloc[0]
        
        return result

class DistanceMetrics:
    """Distance calculation methods for pattern comparison"""
    
    @staticmethod
    def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate normalized Euclidean distance between two arrays
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Normalized Euclidean distance
        """
        if len(x) != len(y):
            raise ValueError("Arrays must have the same length for Euclidean distance")
        
        # Normalize by the sum of values
        x_sum = np.sum(x)
        y_sum = np.sum(y)
        
        if x_sum == 0 or y_sum == 0:
            return np.inf
        
        return np.sqrt(np.square(x - y).sum() / (x_sum * y_sum))
    
    @staticmethod
    def dtw_distance(x: np.ndarray, y: np.ndarray, window: Optional[int] = None) -> float:
        """
        Calculate Dynamic Time Warping distance between two arrays
        
        Args:
            x: First array
            y: Second array
            window: Sakoe-Chiba band width (if None, no constraint)
            
        Returns:
            DTW distance
        """
        n, m = len(x), len(y)
        
        # Initialize DTW matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Apply window constraint if specified
        if window is not None:
            for i in range(n + 1):
                for j in range(m + 1):
                    if abs(i - j) > window:
                        dtw_matrix[i, j] = np.inf
        
        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if window is None or abs(i - j) <= window:
                    cost = abs(x[i-1] - y[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],    # insertion
                        dtw_matrix[i, j-1],    # deletion
                        dtw_matrix[i-1, j-1]   # match
                    )
        
        return dtw_matrix[n, m]
    
    @staticmethod
    def dtw_keogh_lower_bound(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Keogh lower bound for DTW (faster than full DTW)
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Keogh lower bound
        """
        n, m = len(x), len(m)
        
        if n > m:
            x, y = y, x
            n, m = m, n
        
        # Calculate envelope for y
        y_upper = np.maximum.accumulate(y)
        y_lower = np.minimum.accumulate(y)
        
        # Calculate lower bound
        lb = 0
        for i in range(n):
            if x[i] > y_upper[i]:
                lb += (x[i] - y_upper[i]) ** 2
            elif x[i] < y_lower[i]:
                lb += (y_lower[i] - x[i]) ** 2
        
        return np.sqrt(lb)

class PatternRecognizer:
    """Main class for pattern recognition using the above components"""
    
    def __init__(self, threshold_percentage: float = 0.3, 
                 min_duration: str = '5min', timeframe: str = '5min'):
        """
        Initialize the pattern recognizer
        
        Args:
            threshold_percentage: Percentage threshold for pivot points
            min_duration: Minimum duration for movements
            timeframe: Timeframe for linearization
        """
        self.pivot_detector = PivotPointDetector(threshold_percentage, min_duration)
        self.linearizer = PiecewiseLinearizer(timeframe)
        self.distance_metrics = DistanceMetrics()
    
    def extract_pattern(self, df: pd.DataFrame, start_date: Union[str, pd.Timestamp],
                       end_date: Optional[Union[str, pd.Timestamp]] = None,
                       price_col: str = 'close') -> pd.Series:
        """
        Extract a pattern from the given date range
        
        Args:
            df: DataFrame with OHLCV data
            start_date: Start date for pattern
            end_date: End date for pattern
            price_col: Column name for price data
            
        Returns:
            Normalized piecewise linear pattern
        """
        # Find movements
        movements = self.pivot_detector.find_movements(df, price_col)
        
        # Linearize movements
        pattern = self.linearizer.linearize_movements(movements, start_date, end_date)
        
        return pattern
    
    def compare_patterns(self, pattern1: pd.Series, pattern2: pd.Series, 
                        method: str = 'dtw') -> float:
        """
        Compare two patterns using specified distance metric
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            method: Distance method ('euclidean' or 'dtw')
            
        Returns:
            Distance between patterns
        """
        if method == 'euclidean':
            return self.distance_metrics.euclidean_distance(pattern1.values, pattern2.values)
        elif method == 'dtw':
            return self.distance_metrics.dtw_distance(pattern1.values, pattern2.values)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'euclidean' or 'dtw'")
    
    def find_similar_patterns(self, df: pd.DataFrame, target_pattern: pd.Series,
                             threshold: float = 0.1, method: str = 'dtw') -> List[Dict]:
        """
        Find patterns similar to the target pattern
        
        Args:
            df: DataFrame with OHLCV data
            target_pattern: Target pattern to match
            threshold: Maximum distance threshold
            method: Distance method
            
        Returns:
            List of similar patterns with their details
        """
        # This is a simplified version - in practice, you'd want to slide a window
        # over the data and compare each segment
        similar_patterns = []
        
        # For now, return empty list - this would need more sophisticated implementation
        # to slide over the data and find matches
        
        return similar_patterns
