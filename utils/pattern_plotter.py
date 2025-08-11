import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Union
from core.pattern_recognition import PivotPoint, PriceMovement
from core.pattern_clustering import PatternCluster
import warnings

class PatternPlotter:
    """Plotting utility for pattern recognition results"""
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize the pattern plotter
        
        Args:
            theme: Plotly theme to use
        """
        self.theme = theme
    
    def plot_pivot_points(self, df: pd.DataFrame, pivot_points: List[PivotPoint],
                         price_col: str = 'close', title: str = "Pivot Points Detection") -> go.Figure:
        """
        Plot price data with detected pivot points, support/resistance levels, and enhanced styling
        
        Args:
            df: DataFrame with OHLCV data
            pivot_points: List of detected pivot points
            price_col: Column name for price data
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Create subplot with secondary y-axis for volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(title, "Volume Confirmation"),
            row_heights=[0.7, 0.3]
        )
        
        # === STEP 1: Enhanced Pivot Detection ===
        # Use structure builder's swing point detection for more accurate pivots
        from core.structure_builder import detect_swing_points_scipy
        
        # Get swing points using the proven scipy method
        swing_highs_df, swing_lows_df = detect_swing_points_scipy(df, prominence_factor=7.5)
        
        # Convert to our PivotPoint format
        enhanced_pivots = []
        
        # Process swing highs
        for idx, row in swing_highs_df.iterrows():
            # Calculate percentage change from previous low
            prev_low_idx = df.index.get_loc(idx) - 1
            if prev_low_idx >= 0:
                prev_low = df.iloc[prev_low_idx]['Low']
                pct_change = (row['price'] - prev_low) / prev_low
            else:
                pct_change = 0.0
                
            enhanced_pivots.append(PivotPoint(
                timestamp=idx,
                price=row['price'],
                pivot_type='high',
                percentage_change=pct_change,
                start_price=prev_low if prev_low_idx >= 0 else row['price'],
                end_price=row['price']
            ))
        
        # Process swing lows
        for idx, row in swing_lows_df.iterrows():
            # Calculate percentage change from previous high
            prev_high_idx = df.index.get_loc(idx) - 1
            if prev_high_idx >= 0:
                prev_high = df.iloc[prev_high_idx]['High']
                pct_change = (prev_high - row['price']) / prev_high
            else:
                pct_change = 0.0
                
            enhanced_pivots.append(PivotPoint(
                timestamp=idx,
                price=row['price'],
                pivot_type='low',
                percentage_change=pct_change,
                start_price=prev_high if prev_high_idx >= 0 else row['price'],
                end_price=row['price']
            ))
        
        # Sort by timestamp
        enhanced_pivots.sort(key=lambda x: x.timestamp)
        
        # === STEP 2: Volume Confirmation ===
        volume_confirmed_pivots = self._get_volume_confirmed_pivots(df, enhanced_pivots)
        
        # === STEP 3: Support & Resistance Levels ===
        support_resistance = self._calculate_support_resistance_from_pivots(enhanced_pivots)
        
        # Add candlestick data (more professional than line chart)
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'] if 'Open' in df.columns else df[price_col],
            high=df['High'] if 'High' in df.columns else df[price_col],
            low=df['Low'] if 'Low' in df.columns else df[price_col],
            close=df[price_col],
            name='Price Action',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350',
            increasing_fillcolor='#26A69A',
            decreasing_fillcolor='#EF5350',
            line=dict(width=1)
        ), row=1, col=1)
        
        # Add enhanced pivot points with better visibility
        highs = [p for p in enhanced_pivots if p.pivot_type == 'high']
        lows = [p for p in enhanced_pivots if p.pivot_type == 'low']
        
        if highs:
            # Size markers based on percentage change and volume confirmation
            sizes = []
            colors = []
            for p in highs:
                # Larger size for volume-confirmed pivots
                base_size = min(20, max(12, int(p.percentage_change * 2000)))
                if p in volume_confirmed_pivots:
                    sizes.append(base_size + 3)
                    colors.append('#D32F2F')  # Dark red for confirmed
                else:
                    sizes.append(base_size)
                    colors.append('#FF5722')  # Lighter red for unconfirmed
            
            fig.add_trace(go.Scatter(
                x=[p.timestamp for p in highs],
                y=[p.price for p in highs],
                mode='markers+text',
                name='Pivot Highs',
                text=[f"🔴 {p.price:.2f}" for p in highs],
                textposition='top center',
                marker=dict(
                    symbol='diamond',
                    size=sizes,
                    color=colors,
                    line=dict(width=3, color='#B71C1C')
                ),
                textfont=dict(size=10, color='#D32F2F'),
                hoverinfo='text',
                hovertext=[f"<b>Pivot High</b><br>Price: {p.price:.2f}<br>Change: {p.percentage_change:.1%}<br>Volume Confirmed: {'Yes' if p in volume_confirmed_pivots else 'No'}<br>Time: {pd.Timestamp(p.timestamp).strftime('%Y-%m-%d %H:%M')}" 
                          for p in highs]
            ), row=1, col=1)
        
        if lows:
            # Size markers based on percentage change and volume confirmation
            sizes = []
            colors = []
            for p in lows:
                # Larger size for volume-confirmed pivots
                base_size = min(20, max(12, int(p.percentage_change * 2000)))
                if p in volume_confirmed_pivots:
                    sizes.append(base_size + 3)
                    colors.append('#388E3C')  # Dark green for confirmed
                else:
                    sizes.append(base_size)
                    colors.append('#4CAF50')  # Lighter green for unconfirmed
            
            fig.add_trace(go.Scatter(
                x=[p.timestamp for p in lows],
                y=[p.price for p in lows],
                mode='markers+text',
                name='Pivot Lows',
                text=[f"🟢 {p.price:.2f}" for p in lows],
                textposition='bottom center',
                marker=dict(
                    symbol='diamond',
                    size=sizes,
                    color=colors,
                    line=dict(width=3, color='#1B5E20')
                ),
                textfont=dict(size=10, color='#388E3C'),
                hoverinfo='text',
                hovertext=[f"<b>Pivot Low</b><br>Price: {p.price:.2f}<br>Change: {p.percentage_change:.1%}<br>Volume Confirmed: {'Yes' if p in volume_confirmed_pivots else 'No'}<br>Time: {pd.Timestamp(p.timestamp).strftime('%Y-%m-%d %H:%M')}" 
                          for p in lows]
            ), row=1, col=1)
        
        # Add support and resistance levels
        for level_type, levels in support_resistance.items():
            if levels:
                color = '#FF9800' if level_type == 'resistance' else '#2196F3'
                line_style = 'dash' if level_type == 'resistance' else 'dot'
                
                for i, level in enumerate(levels):
                    fig.add_hline(
                        y=level,
                        line_dash=line_style,
                        line_color=color,
                        line_width=2,
                        opacity=0.8,
                        annotation_text=f"{level_type.title()} {i+1}: {level:.2f}",
                        annotation_position="right",
                        annotation_font_size=10,
                        annotation_font_color=color,
                        row=1, col=1
                    )
        
        # Add moving averages
        if len(df) > 20:
            sma_20 = df[price_col].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=sma_20,
                mode='lines',
                name='SMA 20',
                line=dict(color='#FF9800', width=2),
                opacity=0.8
            ), row=1, col=1)
            
            if len(df) > 50:
                sma_50 = df[price_col].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=sma_50,
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='#2196F3', width=2),
                    opacity=0.8
                ), row=1, col=1)
        
        # Add volume with confirmation signals
        if 'Volume' in df.columns:
            # Color volume bars based on pivot confirmation
            volume_colors = []
            for i, (close, open_price) in enumerate(zip(df[price_col], df['Open'] if 'Open' in df.columns else df[price_col])):
                # Check if this timestamp has a volume-confirmed pivot
                timestamp = df.index[i]
                has_confirmed_pivot = any(p.timestamp == timestamp for p in volume_confirmed_pivots)
                
                if has_confirmed_pivot:
                    volume_colors.append('#9C27B0')  # Purple for confirmed pivots
                elif close >= open_price:
                    volume_colors.append('#26A69A')  # Green for bullish
                else:
                    volume_colors.append('#EF5350')  # Red for bearish
            
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            ), row=2, col=1)
            
            # Add volume moving average
            if len(df) > 20:
                volume_sma = df['Volume'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=volume_sma,
                    mode='lines',
                    name='Volume SMA 20',
                    line=dict(color='#9C27B0', width=2),
                    opacity=0.8
                ), row=2, col=1)
                
                # Highlight volume-confirmed pivots
                for pivot in volume_confirmed_pivots:
                    if pivot.timestamp in df.index:
                        fig.add_vrect(
                            x0=pivot.timestamp - pd.Timedelta(hours=1),
                            x1=pivot.timestamp + pd.Timedelta(hours=1),
                            fillcolor="rgba(156, 39, 176, 0.2)",
                            layer="below",
                            line_width=0,
                            row=2, col=1
                        )
        
        # Update layout with professional styling
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20, color='#2E3440')
            ),
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            ),
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color='#2E3440'),
            margin=dict(l=50, r=50, t=80, b=50),
            height=700
        )
        
        # Update axes
        for row in [1, 2]:
            fig.update_xaxes(
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                row=row, col=1
            )
        
        fig.update_yaxes(
            title_text="Price",
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
            range=[df[price_col].min() * 0.995, df[price_col].max() * 1.005],
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Volume",
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
            row=2, col=1
        )
        
        return fig
    
    def _enhance_pivot_detection(self, df: pd.DataFrame, pivot_points: List[PivotPoint], 
                                price_col: str) -> List[PivotPoint]:
        """Enhanced pivot detection with better thresholds and confirmation"""
        if not pivot_points:
            # If no pivot points detected, use enhanced algorithm
            return self._detect_enhanced_pivots(df, price_col)
        
        # Filter pivot points by significance and add confirmation
        enhanced_pivots = []
        prices = df[price_col].values
        timestamps = df.index.values
        
        for pivot in pivot_points:
            try:
                # Convert numpy timestamp to pandas timestamp for get_loc
                pivot_timestamp = pd.Timestamp(pivot.timestamp)
                idx = df.index.get_loc(pivot_timestamp)
                if idx < 5 or idx >= len(prices) - 5:
                    continue
                    
                # Look for confirmation in surrounding candles
                if pivot.pivot_type == 'high':
                    # Check if high is confirmed by surrounding lower highs
                    surrounding_highs = prices[idx-5:idx+6]
                    if pivot.price == max(surrounding_highs):
                        enhanced_pivots.append(pivot)
                else:  # low
                    # Check if low is confirmed by surrounding higher lows
                    surrounding_lows = prices[idx-5:idx+6]
                    if pivot.price == min(surrounding_lows):
                        enhanced_pivots.append(pivot)
            except (KeyError, ValueError):
                # Skip if timestamp not found or conversion fails
                continue
        
        return enhanced_pivots
    
    def _detect_enhanced_pivots(self, df: pd.DataFrame, price_col: str) -> List[PivotPoint]:
        """Enhanced pivot detection algorithm"""
        prices = df[price_col].values
        timestamps = df.index.values
        enhanced_pivots = []
        
        # Use multiple thresholds for different market conditions
        thresholds = [0.001, 0.002, 0.005, 0.01]  # 0.1%, 0.2%, 0.5%, 1%
        
        for threshold in thresholds:
            for i in range(5, len(prices) - 5):
                current_price = prices[i]
                
                # Check for local high
                if (current_price > prices[i-5:i].max() and 
                    current_price > prices[i+1:i+6].max()):
                    
                    # Calculate percentage drop
                    future_drops = (current_price - prices[i:i+10]) / current_price
                    past_drops = (current_price - prices[i-10:i]) / current_price
                    max_drop = max(future_drops.max(), past_drops.max())
                    
                    if max_drop >= threshold:
                        enhanced_pivots.append(PivotPoint(
                            timestamp=timestamps[i],
                            price=current_price,
                            pivot_type='high',
                            percentage_change=max_drop,
                            start_price=prices[i-1],
                            end_price=prices[i+1]
                        ))
                        break
                
                # Check for local low
                elif (current_price < prices[i-5:i].min() and 
                      current_price < prices[i+1:i+6].min()):
                    
                    # Calculate percentage rise
                    future_rises = (prices[i:i+10] - current_price) / current_price
                    past_rises = (prices[i-10:i] - current_price) / current_price
                    max_rise = max(future_rises.max(), past_rises.max())
                    
                    if max_rise >= threshold:
                        enhanced_pivots.append(PivotPoint(
                            timestamp=timestamps[i],
                            price=current_price,
                            pivot_type='low',
                            percentage_change=max_rise,
                            start_price=prices[i-1],
                            end_price=prices[i+1]
                        ))
                        break
        
        return enhanced_pivots
    
    def _calculate_support_resistance(self, df: pd.DataFrame, pivot_points: List[PivotPoint], 
                                    price_col: str) -> Dict[str, List[float]]:
        """Calculate support and resistance levels from pivot points"""
        if not pivot_points:
            return {'support': [], 'resistance': []}
        
        highs = [p.price for p in pivot_points if p.pivot_type == 'high']
        lows = [p.price for p in pivot_points if p.pivot_type == 'low']
        
        # Group nearby levels
        def group_levels(levels, tolerance=0.002):
            if not levels:
                return []
            grouped = []
            levels = sorted(levels)
            
            current_group = [levels[0]]
            for level in levels[1:]:
                if abs(level - current_group[-1]) / current_group[-1] <= tolerance:
                    current_group.append(level)
                else:
                    grouped.append(sum(current_group) / len(current_group))
                    current_group = [level]
            
            if current_group:
                grouped.append(sum(current_group) / len(current_group))
            
            return grouped
        
        resistance = group_levels(highs)
        support = group_levels(lows)
        
        return {
            'resistance': resistance,
            'support': support
        }
    
    def _calculate_trend_lines(self, pivot_points: List[PivotPoint]) -> List[Dict]:
        """Calculate trend lines connecting significant pivot points"""
        if len(pivot_points) < 2:
            return []
        
        trend_lines = []
        sorted_pivots = sorted(pivot_points, key=lambda x: x.timestamp)
        
        for i in range(len(sorted_pivots) - 1):
            p1 = sorted_pivots[i]
            p2 = sorted_pivots[i + 1]
            
            # Calculate slope - handle different timestamp types
            if hasattr(p1.timestamp, 'total_seconds'):
                time_diff = abs((p2.timestamp - p1.timestamp).total_seconds() / 3600)  # hours
            else:
                # Handle numpy timestamps
                time_diff = abs((pd.Timestamp(p2.timestamp) - pd.Timestamp(p1.timestamp)).total_seconds() / 3600)
            
            if time_diff > 0:
                slope = (p2.price - p1.price) / time_diff
                
                trend_lines.append({
                    'start_time': p1.timestamp,
                    'end_time': p2.timestamp,
                    'start_price': p1.price,
                    'end_price': p2.price,
                    'slope': slope,
                    'duration': f"{time_diff:.1f}h"
                })
        
        return trend_lines
    
    def plot_movements(self, df: pd.DataFrame, movements: Dict[str, pd.DataFrame],
                       price_col: str = 'close', title: str = "Price Movements") -> go.Figure:
        """
        Plot price movements between pivot points
        
        Args:
            df: DataFrame with OHLCV data
            movements: Dictionary with 'uptrends' and 'downtrends' DataFrames
            price_col: Column name for price data
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add candlestick data
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[price_col],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=1),
            opacity=0.7
        ))
        
        # Add uptrends
        if not movements['uptrends'].empty:
            for _, movement in movements['uptrends'].iterrows():
                fig.add_trace(go.Scatter(
                    x=[movement['start_timestamp'], movement['end_timestamp']],
                    y=[movement['start_price'], movement['end_price']],
                    mode='lines+markers',
                    name=f"Uptrend {movement['start_timestamp'].strftime('%Y-%m-%d %H:%M')}",
                    line=dict(color='green', width=3),
                    marker=dict(size=8, color='green'),
                    showlegend=False,
                    text=f"Uptrend: {movement['percentage_change']:.1%}<br>Duration: {movement['duration']}",
                    hoverinfo='text'
                ))
        
        # Add downtrends
        if not movements['downtrends'].empty:
            for _, movement in movements['downtrends'].iterrows():
                fig.add_trace(go.Scatter(
                    x=[movement['start_timestamp'], movement['end_timestamp']],
                    y=[movement['start_price'], movement['end_price']],
                    mode='lines+markers',
                    name=f"Downtrend {movement['start_timestamp'].strftime('%Y-%m-%d %H:%M')}",
                    line=dict(color='red', width=3),
                    marker=dict(size=8, color='red'),
                    showlegend=False,
                    text=f"Downtrend: {movement['percentage_change']:.1%}<br>Duration: {movement['duration']}",
                    hoverinfo='text'
                ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Price",
            template=self.theme,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_patterns(self, patterns: List[pd.Series], 
                     pattern_names: Optional[List[str]] = None,
                     title: str = "Pattern Comparison") -> go.Figure:
        """
        Plot multiple patterns for comparison
        
        Args:
            patterns: List of pattern series
            pattern_names: List of names for each pattern
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if pattern_names is None:
            pattern_names = [f"Pattern {i+1}" for i in range(len(patterns))]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, (pattern, name) in enumerate(zip(patterns, pattern_names)):
            if pattern.empty:
                continue
                
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=pattern.index,
                y=pattern.values,
                mode='lines',
                name=name,
                line=dict(color=color, width=2),
                text=f"{name}<br>Value: {pattern.values[-1]:.4f}",
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Normalized Price",
            template=self.theme,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_clusters(self, clusters: Dict[int, PatternCluster],
                     title: str = "Pattern Clusters") -> go.Figure:
        """
        Plot pattern clusters with their centers
        
        Args:
            clusters: Dictionary of pattern clusters
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for cluster_id, cluster in clusters.items():
            color = colors[cluster_id % len(colors)]
            
            # Plot individual patterns in the cluster
            for i, pattern in enumerate(cluster.patterns):
                # Create time index for the pattern
                time_index = pd.RangeIndex(len(pattern))
                
                # Plot with low opacity
                fig.add_trace(go.Scatter(
                    x=time_index,
                    y=pattern,
                    mode='lines',
                    name=f"Cluster {cluster_id} - Pattern {i+1}",
                    line=dict(color=color, width=1, opacity=0.3),
                    showlegend=False
                ))
            
            # Plot cluster center with high opacity
            time_index = pd.RangeIndex(len(cluster.center_pattern))
            fig.add_trace(go.Scatter(
                x=time_index,
                y=cluster.center_pattern,
                mode='lines',
                name=f"Cluster {cluster_id} - Center (Confidence: {cluster.confidence:.3f})",
                line=dict(color=color, width=4, opacity=0.8),
                text=f"Cluster {cluster_id}<br>Confidence: {cluster.confidence:.3f}<br>Patterns: {len(cluster.patterns)}",
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time Steps",
            yaxis_title="Normalized Price",
            template=self.theme,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_pattern_analysis(self, df: pd.DataFrame, 
                            target_pattern: pd.Series,
                            similar_patterns: List[Dict],
                            price_col: str = 'close',
                            title: str = "Pattern Analysis") -> go.Figure:
        """
        Plot pattern analysis showing target pattern and similar matches
        
        Args:
            df: DataFrame with OHLCV data
            target_pattern: Target pattern to match
            similar_patterns: List of similar patterns found
            price_col: Column name for price data
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Price Data with Pattern Matches", "Target Pattern vs Matches"),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # Subplot 1: Price data with matches
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[price_col],
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=1)
            ),
            row=1, col=1
        )
        
        # Add pattern matches
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        for i, match in enumerate(similar_patterns[:5]):  # Show first 5 matches
            color = colors[i % len(colors)]
            
            # This is a simplified visualization - in practice you'd need to
            # map the pattern back to the actual time range in the data
            fig.add_trace(
                go.Scatter(
                    x=[match.get('start_time', df.index[0]), match.get('end_time', df.index[-1])],
                    y=[match.get('start_price', df[price_col].iloc[0]), match.get('end_price', df[price_col].iloc[-1])],
                    mode='markers',
                    name=f"Match {i+1} (Distance: {match.get('distance', 0):.3f})",
                    marker=dict(color=color, size=10, symbol='diamond'),
                    text=f"Match {i+1}<br>Distance: {match.get('distance', 0):.3f}<br>Confidence: {match.get('confidence', 0):.3f}",
                    hoverinfo='text'
                ),
                row=1, col=1
            )
        
        # Subplot 2: Pattern comparison
        fig.add_trace(
            go.Scatter(
                x=target_pattern.index,
                y=target_pattern.values,
                mode='lines',
                name='Target Pattern',
                line=dict(color='black', width=3)
            ),
            row=2, col=1
        )
        
        # Add similar patterns
        for i, match in enumerate(similar_patterns[:3]):  # Show first 3 patterns
            color = colors[i % len(colors)]
            if 'pattern' in match:
                pattern_data = match['pattern']
                if hasattr(pattern_data, 'index') and hasattr(pattern_data, 'values'):
                    fig.add_trace(
                        go.Scatter(
                            x=pattern_data.index,
                            y=pattern_data.values,
                            mode='lines',
                            name=f"Similar Pattern {i+1}",
                            line=dict(color=color, width=2, opacity=0.7)
                        ),
                        row=2, col=1
                    )
        
        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_xaxes(title_text="Time Steps", row=2, col=1)
        fig.update_yaxes(title_text="Normalized Price", row=2, col=1)
        
        return fig
    
    def plot_distance_matrix(self, patterns: List[pd.Series], 
                           distances: np.ndarray,
                           pattern_names: Optional[List[str]] = None,
                           title: str = "Pattern Distance Matrix") -> go.Figure:
        """
        Plot distance matrix between patterns
        
        Args:
            patterns: List of pattern series
            distances: Distance matrix (n x n)
            pattern_names: List of names for each pattern
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if pattern_names is None:
            pattern_names = [f"Pattern {i+1}" for i in range(len(patterns))]
        
        fig = go.Figure(data=go.Heatmap(
            z=distances,
            x=pattern_names,
            y=pattern_names,
            colorscale='Viridis',
            text=[[f"{distances[i][j]:.3f}" for j in range(len(distances[i]))] 
                  for i in range(len(distances))],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Patterns",
            yaxis_title="Patterns",
            template=self.theme
        )
        
        return fig

    def _get_volume_confirmed_pivots(self, df: pd.DataFrame, pivot_points: List[PivotPoint]) -> List[PivotPoint]:
        """
        Get pivot points that are confirmed by high volume
        
        Args:
            df: DataFrame with OHLCV data
            pivot_points: List of detected pivot points
            
        Returns:
            List of volume-confirmed pivot points
        """
        if 'Volume' not in df.columns:
            return []
        
        # Calculate volume moving average
        volume_sma = df['Volume'].rolling(window=20).mean()
        volume_threshold = volume_sma * 1.5  # 50% above average
        
        confirmed_pivots = []
        for pivot in pivot_points:
            if pivot.timestamp in df.index:
                pivot_idx = df.index.get_loc(pivot.timestamp)
                if pivot_idx < len(df):
                    pivot_volume = df.iloc[pivot_idx]['Volume']
                    avg_volume = volume_sma.iloc[pivot_idx] if not pd.isna(volume_sma.iloc[pivot_idx]) else 0
                    
                    if pivot_volume > avg_volume * 1.5:
                        confirmed_pivots.append(pivot)
        
        return confirmed_pivots
    
    def _calculate_support_resistance_from_pivots(self, pivot_points: List[PivotPoint]) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels from pivot points
        
        Args:
            pivot_points: List of detected pivot points
            
        Returns:
            Dictionary with 'support' and 'resistance' lists
        """
        if not pivot_points:
            return {'support': [], 'resistance': []}
        
        # Group pivot points by type
        highs = [p.price for p in pivot_points if p.pivot_type == 'high']
        lows = [p.price for p in pivot_points if p.pivot_type == 'low']
        
        # Group similar levels (within 0.2% tolerance)
        def group_levels(levels, tolerance=0.002):
            if not levels:
                return []
            
            grouped = []
            levels = sorted(levels)
            
            current_group = [levels[0]]
            for level in levels[1:]:
                if abs(level - current_group[-1]) / current_group[-1] <= tolerance:
                    current_group.append(level)
                else:
                    # Calculate average of current group
                    grouped.append(sum(current_group) / len(current_group))
                    current_group = [level]
            
            # Don't forget the last group
            if current_group:
                grouped.append(sum(current_group) / len(current_group))
            
            return grouped
        
        resistance_levels = group_levels(highs)
        support_levels = group_levels(lows)
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def _calculate_rsi(self, df: pd.DataFrame, price_col: str, period: int = 14) -> Optional[pd.Series]:
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price data
            period: RSI period (default 14)
            
        Returns:
            RSI series or None if calculation fails
        """
        try:
            # Calculate price changes
            delta = df[price_col].diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            print(f"RSI calculation error: {e}")
            return None
    
    def _detect_rsi_divergence(self, df: pd.DataFrame, pivot_points: List[PivotPoint], 
                              rsi_data: pd.Series) -> List[Dict]:
        """
        Detect RSI divergence patterns
        
        Args:
            df: DataFrame with price data
            pivot_points: List of detected pivot points
            rsi_data: RSI series
            
        Returns:
            List of divergence signals
        """
        if rsi_data is None or len(pivot_points) < 4:
            return []
        
        divergence_signals = []
        
        # Get recent pivot points (last 20 to avoid too many signals)
        recent_pivots = sorted(pivot_points, key=lambda x: x.timestamp)[-20:]
        
        # Find price highs and lows
        price_highs = [p for p in recent_pivots if p.pivot_type == 'high']
        price_lows = [p for p in recent_pivots if p.pivot_type == 'low']
        
        # Need at least 2 highs and 2 lows for divergence
        if len(price_highs) >= 2 and len(price_lows) >= 2:
            # Bearish divergence: Price makes higher high, RSI makes lower high
            if len(price_highs) >= 2:
                high1, high2 = price_highs[-2], price_highs[-1]
                
                if high2.price > high1.price:  # Price makes higher high
                    # Get RSI values at these times
                    rsi1 = rsi_data.loc[high1.timestamp] if high1.timestamp in rsi_data.index else None
                    rsi2 = rsi_data.loc[high2.timestamp] if high2.timestamp in rsi_data.index else None
                    
                    if rsi1 is not None and rsi2 is not None and rsi2 < rsi1:
                        divergence_signals.append({
                            'type': 'bearish',
                            'price_high1': high1.timestamp,
                            'price_high2': high2.timestamp,
                            'rsi_high1': rsi1,
                            'rsi_high2': rsi2,
                            'strength': abs(rsi1 - rsi2)
                        })
            
            # Bullish divergence: Price makes lower low, RSI makes higher low
            if len(price_lows) >= 2:
                low1, low2 = price_lows[-2], price_lows[-1]
                
                if low2.price < low1.price:  # Price makes lower low
                    # Get RSI values at these times
                    rsi1 = rsi_data.loc[low1.timestamp] if low1.timestamp in rsi_data.index else None
                    rsi2 = rsi_data.loc[low2.timestamp] if low2.timestamp in rsi_data.index else None
                    
                    if rsi1 is not None and rsi2 is not None and rsi2 > rsi1:
                        divergence_signals.append({
                            'type': 'bullish',
                            'price_low1': low1.timestamp,
                            'price_low2': low2.timestamp,
                            'rsi_low1': rsi1,
                            'rsi_low2': rsi2,
                            'strength': abs(rsi2 - rsi1)
                        })
        
        return divergence_signals

    def plot_comprehensive_pivot_analysis(self, df: pd.DataFrame, pivot_points: List[PivotPoint],
                                        price_col: str = 'close', title: str = "Comprehensive Pivot Analysis") -> go.Figure:
        """
        Create a comprehensive pivot analysis chart with all features:
        - Actual pivot points detection and plotting
        - Volume confirmation linked to pivots
        - Support & resistance levels from pivots
        - RSI divergence for high accuracy
        
        Args:
            df: DataFrame with OHLCV data
            pivot_points: List of detected pivot points
            price_col: Column name for price data
            title: Chart title
            
        Returns:
            Plotly figure with comprehensive analysis
        """
        # Create subplot with multiple panels
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(title, "Volume Confirmation", "RSI Divergence", "Pivot Strength"),
            row_heights=[0.5, 0.2, 0.2, 0.1]
        )
        
        # === STEP 1: Enhanced Pivot Detection ===
        # Use structure builder's swing point detection for more accurate pivots
        from core.structure_builder import detect_swing_points_scipy
        
        # Get swing points using the proven scipy method
        swing_highs_df, swing_lows_df = detect_swing_points_scipy(df, prominence_factor=7.5)
        
        # Convert to our PivotPoint format
        enhanced_pivots = []
        
        # Process swing highs
        for idx, row in swing_highs_df.iterrows():
            # Calculate percentage change from previous low
            prev_low_idx = df.index.get_loc(idx) - 1
            if prev_low_idx >= 0:
                prev_low = df.iloc[prev_low_idx]['Low']
                pct_change = (row['price'] - prev_low) / prev_low
            else:
                pct_change = 0.0
                
            enhanced_pivots.append(PivotPoint(
                timestamp=idx,
                price=row['price'],
                pivot_type='high',
                percentage_change=pct_change,
                start_price=prev_low if prev_low_idx >= 0 else row['price'],
                end_price=row['price']
            ))
        
        # Process swing lows
        for idx, row in swing_lows_df.iterrows():
            # Calculate percentage change from previous high
            prev_high_idx = df.index.get_loc(idx) - 1
            if prev_high_idx >= 0:
                prev_high = df.iloc[prev_high_idx]['High']
                pct_change = (prev_high - row['price']) / prev_high
            else:
                pct_change = 0.0
                
            enhanced_pivots.append(PivotPoint(
                timestamp=idx,
                price=row['price'],
                pivot_type='low',
                percentage_change=pct_change,
                start_price=prev_high if prev_high_idx >= 0 else row['price'],
                end_price=row['price']
            ))
        
        # Sort by timestamp
        enhanced_pivots.sort(key=lambda x: x.timestamp)
        
        # === STEP 2: Volume Confirmation ===
        volume_confirmed_pivots = self._get_volume_confirmed_pivots(df, enhanced_pivots)
        
        # === STEP 3: Support & Resistance Levels ===
        support_resistance = self._calculate_support_resistance_from_pivots(enhanced_pivots)
        
        # === STEP 4: RSI Calculation and Divergence Detection ===
        rsi_data = self._calculate_rsi(df, price_col)
        divergence_signals = self._detect_rsi_divergence(df, enhanced_pivots, rsi_data)
        
        # === STEP 5: Plot Price Action with Pivots ===
        # Add candlestick data
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'] if 'Open' in df.columns else df[price_col],
            high=df['High'] if 'High' in df.columns else df[price_col],
            low=df['Low'] if 'Low' in df.columns else df[price_col],
            close=df[price_col],
            name='Price Action',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350',
            increasing_fillcolor='#26A69A',
            decreasing_fillcolor='#EF5350',
            line=dict(width=1)
        ), row=1, col=1)
        
        # === STEP 6: Plot Actual Pivot Points ===
        highs = [p for p in enhanced_pivots if p.pivot_type == 'high']
        lows = [p for p in enhanced_pivots if p.pivot_type == 'low']
        
        if highs:
            # Size markers based on percentage change and volume confirmation
            sizes = []
            colors = []
            for p in highs:
                # Larger size for volume-confirmed pivots
                base_size = min(25, max(15, int(p.percentage_change * 2000)))
                if p in volume_confirmed_pivots:
                    sizes.append(base_size + 5)
                    colors.append('#D32F2F')  # Dark red for confirmed
                else:
                    sizes.append(base_size)
                    colors.append('#FF5722')  # Lighter red for unconfirmed
            
            fig.add_trace(go.Scatter(
                x=[p.timestamp for p in highs],
                y=[p.price for p in highs],
                mode='markers+text',
                name='Pivot Highs',
                text=[f"🔴 {p.price:.2f}" for p in highs],
                textposition='top center',
                marker=dict(
                    symbol='diamond',
                    size=sizes,
                    color=colors,
                    line=dict(width=3, color='#B71C1C')
                ),
                textfont=dict(size=10, color='#D32F2F'),
                hoverinfo='text',
                hovertext=[f"<b>Pivot High</b><br>Price: {p.price:.2f}<br>Change: {p.percentage_change:.1%}<br>Volume Confirmed: {'Yes' if p in volume_confirmed_pivots else 'No'}<br>Time: {pd.Timestamp(p.timestamp).strftime('%Y-%m-%d %H:%M')}" 
                          for p in highs]
            ), row=1, col=1)
        
        if lows:
            # Size markers based on percentage change and volume confirmation
            sizes = []
            colors = []
            for p in lows:
                # Larger size for volume-confirmed pivots
                base_size = min(25, max(15, int(p.percentage_change * 2000)))
                if p in volume_confirmed_pivots:
                    sizes.append(base_size + 5)
                    colors.append('#388E3C')  # Dark green for confirmed
                else:
                    sizes.append(base_size)
                    colors.append('#4CAF50')  # Lighter green for unconfirmed
            
            fig.add_trace(go.Scatter(
                x=[p.timestamp for p in lows],
                y=[p.price for p in lows],
                mode='markers+text',
                name='Pivot Lows',
                text=[f"🟢 {p.price:.2f}" for p in lows],
                textposition='bottom center',
                marker=dict(
                    symbol='diamond',
                    size=sizes,
                    color=colors,
                    line=dict(width=3, color='#1B5E20')
                ),
                textfont=dict(size=10, color='#388E3C'),
                hoverinfo='text',
                hovertext=[f"<b>Pivot Low</b><br>Price: {p.price:.2f}<br>Change: {p.percentage_change:.1%}<br>Volume Confirmed: {'Yes' if p in volume_confirmed_pivots else 'No'}<br>Time: {pd.Timestamp(p.timestamp).strftime('%Y-%m-%d %H:%M')}" 
                          for p in lows]
            ), row=1, col=1)
        
        # === STEP 7: Plot Support & Resistance Levels ===
        # Only plot S/R levels for volume-confirmed pivots to reduce clutter
        volume_confirmed_sr = self._calculate_support_resistance_from_pivots(volume_confirmed_pivots)
        
        for level_type, levels in volume_confirmed_sr.items():
            if levels:
                color = '#FF9800' if level_type == 'resistance' else '#2196F3'
                line_style = 'dash' if level_type == 'resistance' else 'dot'
                
                for i, level in enumerate(levels):
                    fig.add_hline(
                        y=level,
                        line_dash=line_style,
                        line_color=color,
                        line_width=2,
                        opacity=0.8,
                        annotation_text=f"{level_type.title()} {i+1}: {level:.2f}",
                        annotation_position="right",
                        annotation_font_size=10,
                        annotation_font_color=color,
                        row=1, col=1
                    )
        
        # === STEP 8: Plot Volume with Confirmation ===
        if 'Volume' in df.columns:
            # Color volume bars based on pivot confirmation
            volume_colors = []
            for i, (close, open_price) in enumerate(zip(df[price_col], df['Open'] if 'Open' in df.columns else df[price_col])):
                # Check if this timestamp has a volume-confirmed pivot
                timestamp = df.index[i]
                has_confirmed_pivot = any(p.timestamp == timestamp for p in volume_confirmed_pivots)
                
                if has_confirmed_pivot:
                    volume_colors.append('#9C27B0')  # Purple for confirmed pivots
                elif close >= open_price:
                    volume_colors.append('#26A69A')  # Green for bullish
                else:
                    volume_colors.append('#EF5350')  # Red for bearish
            
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            ), row=2, col=1)
            
            # Add volume moving average
            if len(df) > 20:
                volume_sma = df['Volume'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=volume_sma,
                    mode='lines',
                    name='Volume SMA 20',
                    line=dict(color='#9C27B0', width=2),
                    opacity=0.8
                ), row=2, col=1)
                
                # Highlight volume-confirmed pivots
                for pivot in volume_confirmed_pivots:
                    if pivot.timestamp in df.index:
                        fig.add_vrect(
                            x0=pivot.timestamp - pd.Timedelta(hours=1),
                            x1=pivot.timestamp + pd.Timedelta(hours=1),
                            fillcolor="rgba(156, 39, 176, 0.2)",
                            layer="below",
                            line_width=0,
                            row=2, col=1
                        )
        
        # === STEP 9: Plot RSI with Divergence ===
        if rsi_data is not None:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=rsi_data,
                mode='lines',
                name='RSI (14)',
                line=dict(color='#E91E63', width=2),
                opacity=0.8
            ), row=3, col=1)
            
            # Add RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="#FF9800", line_width=1, opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#FF9800", line_width=1, opacity=0.5, row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="#9E9E9E", line_width=1, opacity=0.3, row=3, col=1)
            
            # Plot divergence signals
            for signal in divergence_signals:
                if signal['type'] == 'bearish':
                    # Connect the two highs with a line on RSI pane
                    fig.add_trace(go.Scatter(
                        x=[signal['price_high1'], signal['price_high2']],
                        y=[signal['rsi_high1'], signal['rsi_high2']],
                        mode='lines+markers',
                        name='Bearish Divergence',
                        line=dict(color='#F44336', width=3, dash='dash'),
                        marker=dict(symbol='x', size=12, color='#F44336'),
                        showlegend=False
                    ), row=3, col=1)
                    
                    # Connect the two highs with a line on price pane
                    fig.add_trace(go.Scatter(
                        x=[signal['price_high1'], signal['price_high2']],
                        y=[signal['price_high1'], signal['price_high2']],
                        mode='lines+markers',
                        name='Bearish Divergence (Price)',
                        line=dict(color='#F44336', width=3, dash='dash'),
                        marker=dict(symbol='x', size=12, color='#F44336'),
                        showlegend=False
                    ), row=1, col=1)
                    
                    # Add annotation on RSI pane
                    fig.add_annotation(
                        x=signal['price_high2'],
                        y=signal['rsi_high2'],
                        text="🔴 Bearish Divergence",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='#F44336',
                        ax=0,
                        ay=-40,
                        font=dict(size=12, color='#F44336'),
                        row=3, col=1
                    )
                    
                elif signal['type'] == 'bullish':
                    # Connect the two lows with a line on RSI pane
                    fig.add_trace(go.Scatter(
                        x=[signal['price_low1'], signal['price_low2']],
                        y=[signal['rsi_low1'], signal['rsi_low2']],
                        mode='lines+markers',
                        name='Bullish Divergence',
                        line=dict(color='#4CAF50', width=3, dash='dash'),
                        marker=dict(symbol='x', size=12, color='#4CAF50'),
                        showlegend=False
                    ), row=3, col=1)
                    
                    # Connect the two lows with a line on price pane
                    fig.add_trace(go.Scatter(
                        x=[signal['price_low1'], signal['price_low2']],
                        y=[signal['price_low1'], signal['price_low2']],
                        mode='lines+markers',
                        name='Bullish Divergence (Price)',
                        line=dict(color='#4CAF50', width=3, dash='dash'),
                        marker=dict(symbol='x', size=12, color='#4CAF50'),
                        showlegend=False
                    ), row=1, col=1)
                    
                    # Add annotation on RSI pane
                    fig.add_annotation(
                        x=signal['price_low2'],
                        y=signal['rsi_low2'],
                        text="🟢 Bullish Divergence",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='#4CAF50',
                        ax=0,
                        ay=40,
                        font=dict(size=12, color='#4CAF50'),
                        row=3, col=1
                    )
        
        # === STEP 10: Plot Pivot Strength ===
        pivot_strength = self._calculate_pivot_strength(df, enhanced_pivots, price_col)
        if pivot_strength:
            strength_values = [pivot_strength.get(ts, 0) for ts in df.index]
            fig.add_trace(go.Scatter(
                x=df.index,
                y=strength_values,
                mode='lines',
                name='Pivot Strength',
                line=dict(color='#E91E63', width=2),
                fill='tonexty',
                fillcolor='rgba(233, 30, 99, 0.1)'
            ), row=4, col=1)
        
        # === STEP 11: Update Layout ===
        # Calculate dynamic price range to prevent axis distortion
        price_range = df[price_col].max() - df[price_col].min()
        price_padding = price_range * 0.05  # 5% padding
        price_min = df[price_col].min() - price_padding
        price_max = df[price_col].max() + price_padding
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=22, color='#2E3440')
            ),
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            ),
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color='#2E3440'),
            margin=dict(l=50, r=50, t=100, b=50),
            height=1000
        )
        
        # Update axes
        for row in [1, 2, 3, 4]:
            fig.update_xaxes(
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                row=row, col=1
            )
        
        fig.update_yaxes(
            title_text="Price",
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
            range=[price_min, price_max],
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Volume",
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="RSI",
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
            range=[0, 100],
            row=3, col=1
        )
        fig.update_yaxes(
            title_text="Strength",
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
            row=4, col=1
        )
        
        return fig
    
    def _calculate_pivot_strength(self, df: pd.DataFrame, pivot_points: List[PivotPoint], 
                                 price_col: str) -> Dict[pd.Timestamp, float]:
        """Calculate strength metrics for pivot points"""
        if not pivot_points:
            return {}
        
        strength_dict = {}
        prices = df[price_col].values
        timestamps = df.index.values
        
        for pivot in pivot_points:
            try:
                # Convert numpy timestamp to pandas timestamp for get_loc
                pivot_timestamp = pd.Timestamp(pivot.timestamp)
                idx = df.index.get_loc(pivot_timestamp)
                if idx < 10 or idx >= len(prices) - 10:
                    continue
                
                # Calculate strength based on multiple factors
                strength = 0.0
                
                # Factor 1: Percentage change (higher is stronger)
                strength += min(1.0, pivot.percentage_change * 10)
                
                # Factor 2: Volume confirmation (if available)
                if 'Volume' in df.columns:
                    volume_idx = min(idx, len(df) - 1)
                    avg_volume = df['Volume'].rolling(window=20).mean().iloc[volume_idx]
                    current_volume = df['Volume'].iloc[volume_idx]
                    if avg_volume > 0:
                        volume_factor = min(1.0, current_volume / avg_volume)
                        strength += volume_factor * 0.3
                
                # Factor 3: Price distance from moving averages
                if len(df) > 20:
                    sma_20 = df[price_col].rolling(window=20).mean().iloc[idx]
                    if sma_20 > 0:
                        ma_distance = abs(pivot.price - sma_20) / sma_20
                        strength += min(1.0, ma_distance * 5)
                
                # Factor 4: Time confirmation (longer confirmation period = stronger)
                strength += min(0.5, 5 / 10)  # Use fixed value since confirmation_periods not available here
                
                strength_dict[pivot_timestamp] = min(1.0, strength)
                
            except (KeyError, IndexError, ValueError):
                continue
        
        return strength_dict
