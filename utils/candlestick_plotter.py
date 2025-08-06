import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
from core.candlestick_patterns import CandlestickPattern, PatternType, CandlestickPatternDetector

class CandlestickPatternPlotter:
    """
    Advanced candlestick pattern plotting with professional styling
    """
    
    def __init__(self):
        self.pattern_colors = {
            PatternType.BULLISH_ENGULFING: '#00E676',
            PatternType.BEARISH_ENGULFING: '#FF1744',
            PatternType.HAMMER: '#4CAF50',
            PatternType.HANGING_MAN: '#FF5722',
            PatternType.INVERTED_HAMMER: '#2196F3',
            PatternType.SHOOTING_STAR: '#FF9800',
            PatternType.DOJI: '#9C27B0',
            PatternType.SPINNING_TOP: '#673AB7',
            PatternType.MORNING_STAR: '#00BCD4',
            PatternType.EVENING_STAR: '#E91E63',
            PatternType.THREE_WHITE_SOLDIERS: '#8BC34A',
            PatternType.THREE_BLACK_CROWS: '#F44336',
            PatternType.TWEEZER_TOP: '#FF6F00',
            PatternType.TWEEZER_BOTTOM: '#388E3C'
        }
        
        self.pattern_symbols = {
            PatternType.BULLISH_ENGULFING: 'triangle-up',
            PatternType.BEARISH_ENGULFING: 'triangle-down',
            PatternType.HAMMER: 'circle',
            PatternType.HANGING_MAN: 'circle-open',
            PatternType.INVERTED_HAMMER: 'diamond',
            PatternType.SHOOTING_STAR: 'diamond-open',
            PatternType.DOJI: 'cross',
            PatternType.SPINNING_TOP: 'cross-open',
            PatternType.MORNING_STAR: 'star',
            PatternType.EVENING_STAR: 'star-open',
            PatternType.THREE_WHITE_SOLDIERS: 'triangle-up-open',
            PatternType.THREE_BLACK_CROWS: 'triangle-down-open',
            PatternType.TWEEZER_TOP: 'square',
            PatternType.TWEEZER_BOTTOM: 'square-open'
        }

    def _create_base_candlestick_chart(self, df: pd.DataFrame, title: str, symbol: str) -> go.Figure:
        """Create base candlestick chart"""
        fig = go.Figure(data=go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350'
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=900,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Time",
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title="Price",
                gridcolor='rgba(128,128,128,0.2)'
            )
        )
        
        return fig

    def plot_patterns_overview(self, df: pd.DataFrame, patterns: List[CandlestickPattern], 
                              symbol: str, min_confidence: float = 0.6) -> go.Figure:
        """
        Create a comprehensive chart showing all detected candlestick patterns
        """
        fig = self._create_base_candlestick_chart(df, f"{symbol} - Candlestick Patterns Overview", symbol)
        
        # Filter patterns by confidence
        filtered_patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        # Group patterns by type for statistics
        pattern_counts = {}
        
        for pattern in filtered_patterns:
            pattern_type = pattern.pattern_type
            
            # Count patterns
            if pattern_type not in pattern_counts:
                pattern_counts[pattern_type] = 0
            pattern_counts[pattern_type] += 1
            
            # Get color and symbol
            color = self.pattern_colors.get(pattern_type, '#FFFFFF')
            symbol_shape = self.pattern_symbols.get(pattern_type, 'circle')
            
            # Add pattern marker
            fig.add_trace(go.Scatter(
                x=[pattern.timestamp],
                y=[pattern.price],
                mode='markers',
                marker=dict(
                    symbol=symbol_shape,
                    size=15,
                    color=color,
                    line=dict(width=2, color='white'),
                    opacity=0.9
                ),
                name=pattern.pattern_type.value,
                hovertemplate=self._get_pattern_hover_template(pattern),
                showlegend=False
            ))
            
            # Add pattern label
            fig.add_annotation(
                x=pattern.timestamp,
                y=pattern.price,
                text=f"<b>{pattern.pattern_type.value}</b><br>{pattern.confidence:.0%}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                ax=0,
                ay=-40 if pattern.direction == 'Bullish' else 40,
                bgcolor=color,
                font=dict(color='white', size=9, weight="bold"),
                borderpad=4,
                bordercolor='white',
                borderwidth=1,
                opacity=0.9
            )
        
        # Add legend and statistics
        self._add_pattern_legend(fig, df)
        self._add_pattern_statistics(fig, df, filtered_patterns, pattern_counts)
        
        return fig

    def plot_pattern_details(self, df: pd.DataFrame, patterns: List[CandlestickPattern], 
                            symbol: str, pattern_type: PatternType = None) -> go.Figure:
        """
        Create detailed chart for specific pattern type
        """
        if pattern_type:
            filtered_patterns = [p for p in patterns if p.pattern_type == pattern_type]
            title = f"{symbol} - {pattern_type.value} Patterns"
        else:
            filtered_patterns = patterns
            title = f"{symbol} - All Candlestick Patterns"
        
        fig = self._create_base_candlestick_chart(df, title, symbol)
        
        for pattern in filtered_patterns:
            color = self.pattern_colors.get(pattern.pattern_type, '#FFFFFF')
            symbol_shape = self.pattern_symbols.get(pattern.pattern_type, 'circle')
            
            # Highlight the candles involved in the pattern
            for candle_idx in pattern.candle_indices:
                if candle_idx < len(df):
                    candle_time = df.index[candle_idx]
                    candle_high = df.iloc[candle_idx]['High']
                    candle_low = df.iloc[candle_idx]['Low']
                    
                    # Add highlighting box around the pattern candles
                    fig.add_shape(
                        type="rect",
                        x0=candle_time,
                        y0=candle_low * 0.999,
                        x1=candle_time,
                        y1=candle_high * 1.001,
                        line=dict(color=color, width=3),
                        fillcolor=color,
                        opacity=0.2,
                        layer="below"
                    )
            
            # Add pattern marker
            fig.add_trace(go.Scatter(
                x=[pattern.timestamp],
                y=[pattern.price],
                mode='markers',
                marker=dict(
                    symbol=symbol_shape,
                    size=18,
                    color=color,
                    line=dict(width=3, color='white'),
                    opacity=0.95
                ),
                name=pattern.pattern_type.value,
                hovertemplate=self._get_detailed_hover_template(pattern),
                showlegend=False
            ))
        
        # Add detailed legend
        self._add_detailed_legend(fig, df, pattern_type)
        
        return fig

    def plot_pattern_analysis(self, df: pd.DataFrame, patterns: List[CandlestickPattern], 
                             symbol: str) -> go.Figure:
        """
        Create analysis chart with pattern frequency and performance
        """
        # Create subplot with candlestick and pattern frequency
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(f"{symbol} - Price Action", "Pattern Frequency"),
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1)
        
        # Add patterns to main chart
        for pattern in patterns:
            color = self.pattern_colors.get(pattern.pattern_type, '#FFFFFF')
            symbol_shape = self.pattern_symbols.get(pattern.pattern_type, 'circle')
            
            fig.add_trace(go.Scatter(
                x=[pattern.timestamp],
                y=[pattern.price],
                mode='markers',
                marker=dict(
                    symbol=symbol_shape,
                    size=12,
                    color=color,
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                name=pattern.pattern_type.value,
                hovertemplate=self._get_pattern_hover_template(pattern),
                showlegend=False
            ), row=1, col=1)
        
        # Create pattern frequency data
        pattern_dates = [p.timestamp for p in patterns]
        if pattern_dates:
            # Group patterns by date
            pattern_df = pd.DataFrame(pattern_dates, columns=['date'])
            pattern_df['count'] = 1
            daily_patterns = pattern_df.groupby(pattern_df['date'].dt.date)['count'].sum()
            
            # Add frequency bar chart
            fig.add_trace(go.Bar(
                x=daily_patterns.index,
                y=daily_patterns.values,
                name="Patterns per Day",
                marker_color='#FFA726',
                opacity=0.7
            ), row=2, col=1)
        
        fig.update_layout(
            template="plotly_dark",
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig

    def plot_bullish_patterns(self, df: pd.DataFrame, patterns: List[CandlestickPattern], 
                             symbol: str) -> go.Figure:
        """Create chart showing only bullish patterns"""
        bullish_patterns = [p for p in patterns if p.direction == 'Bullish']
        fig = self._create_base_candlestick_chart(df, f"{symbol} - Bullish Candlestick Patterns", symbol)
        
        for pattern in bullish_patterns:
            color = self.pattern_colors.get(pattern.pattern_type, '#00E676')
            symbol_shape = self.pattern_symbols.get(pattern.pattern_type, 'triangle-up')
            
            fig.add_trace(go.Scatter(
                x=[pattern.timestamp],
                y=[pattern.price],
                mode='markers',
                marker=dict(
                    symbol=symbol_shape,
                    size=16,
                    color=color,
                    line=dict(width=2, color='white'),
                    opacity=0.9
                ),
                name=pattern.pattern_type.value,
                hovertemplate=self._get_pattern_hover_template(pattern),
                showlegend=False
            ))
        
        self._add_bullish_legend(fig, df)
        return fig

    def plot_bearish_patterns(self, df: pd.DataFrame, patterns: List[CandlestickPattern], 
                             symbol: str) -> go.Figure:
        """Create chart showing only bearish patterns"""
        bearish_patterns = [p for p in patterns if p.direction == 'Bearish']
        fig = self._create_base_candlestick_chart(df, f"{symbol} - Bearish Candlestick Patterns", symbol)
        
        for pattern in bearish_patterns:
            color = self.pattern_colors.get(pattern.pattern_type, '#FF1744')
            symbol_shape = self.pattern_symbols.get(pattern.pattern_type, 'triangle-down')
            
            fig.add_trace(go.Scatter(
                x=[pattern.timestamp],
                y=[pattern.price],
                mode='markers',
                marker=dict(
                    symbol=symbol_shape,
                    size=16,
                    color=color,
                    line=dict(width=2, color='white'),
                    opacity=0.9
                ),
                name=pattern.pattern_type.value,
                hovertemplate=self._get_pattern_hover_template(pattern),
                showlegend=False
            ))
        
        self._add_bearish_legend(fig, df)
        return fig

    def _get_pattern_hover_template(self, pattern: CandlestickPattern) -> str:
        """Get hover template for pattern"""
        template = f"<b>📊 {pattern.pattern_type.value}</b><br>"
        template += f"<b>Direction: {pattern.direction}</b><br>"
        template += f"<b>Price: ${pattern.price:.2f}</b><br>"
        template += f"Confidence: {pattern.confidence:.0%}<br>"
        template += f"Candles: {len(pattern.candle_indices)}<br>"
        template += f"Time: {pattern.timestamp}<br>"
        template += f"<i>{pattern.description}</i><extra></extra>"
        return template

    def _get_detailed_hover_template(self, pattern: CandlestickPattern) -> str:
        """Get detailed hover template for pattern"""
        template = f"<b>📊 {pattern.pattern_type.value}</b><br>"
        template += f"<b>Direction: {pattern.direction}</b><br>"
        template += f"<b>Price: ${pattern.price:.2f}</b><br>"
        template += f"<b>Confidence: {pattern.confidence:.1%}</b><br>"
        template += f"Candles Involved: {len(pattern.candle_indices)}<br>"
        template += f"Indices: {pattern.candle_indices}<br>"
        template += f"Timestamp: {pattern.timestamp}<br>"
        template += f"<br><i>{pattern.description}</i><extra></extra>"
        return template

    def _add_pattern_legend(self, fig: go.Figure, df: pd.DataFrame):
        """Add comprehensive pattern legend"""
        legend_text = """<b>📊 Professional Candlestick Patterns</b>

<b>Single Candle (15M/1H optimal):</b>
• Hammer/Hanging Man - BOS/CHOCH entries
• Shooting Star/Inverted Hammer - Reversal signals
• Doji/Spinning Top - Entry warnings

<b>2-Candle (15M/1H/4H):</b>
• Engulfing - Strong reversals
• Tweezer Top/Bottom - Strong rejections

<b>3-Candle (1H/4H/1D):</b>
• Morning/Evening Star - High confluence
• Three Soldiers/Crows - Trend confirmation

<i>Timeframe-optimized confidence scoring</i>"""

        fig.add_annotation(
            x=df.index[8] if len(df) > 8 else df.index[0],
            y=df['High'].max() * 0.996,
            text=legend_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color='white', size=10, family="Arial"),
            bgcolor='rgba(15,15,15,0.95)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1,
            borderpad=12
        )

    def _add_pattern_statistics(self, fig: go.Figure, df: pd.DataFrame, 
                               patterns: List[CandlestickPattern], pattern_counts: Dict):
        """Add pattern statistics"""
        total_patterns = len(patterns)
        bullish_patterns = len([p for p in patterns if p.direction == 'Bullish'])
        bearish_patterns = len([p for p in patterns if p.direction == 'Bearish'])
        avg_confidence = sum(p.confidence for p in patterns) / total_patterns if total_patterns > 0 else 0

        stats_text = f"""<b>Pattern Statistics</b>
Total Patterns: {total_patterns}
📈 Bullish: {bullish_patterns}
📉 Bearish: {bearish_patterns}
Avg Confidence: {avg_confidence:.0%}

<b>Most Common:</b>"""

        # Add top 3 most common patterns
        sorted_counts = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for pattern_type, count in sorted_counts:
            stats_text += f"\n• {pattern_type.value}: {count}"

        fig.add_annotation(
            x=df.index[-8] if len(df) > 8 else df.index[-1],
            y=df['Low'].min() * 1.004,
            text=stats_text,
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color='white', size=9, family="Arial"),
            bgcolor='rgba(15,15,15,0.9)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1,
            borderpad=10
        )

    def _add_detailed_legend(self, fig: go.Figure, df: pd.DataFrame, pattern_type: PatternType = None):
        """Add detailed legend for specific pattern"""
        if pattern_type:
            legend_text = f"""<b>📊 {pattern_type.value} Pattern</b>

<b>Pattern Details:</b>
{self._get_pattern_description(pattern_type)}

<b>Trading Signals:</b>
{self._get_pattern_signals(pattern_type)}

<i>Highlighted candles show pattern formation</i>"""
        else:
            legend_text = """<b>📊 All Candlestick Patterns</b>

<b>Pattern Analysis:</b>
• Reversal patterns indicate trend changes
• Confirmation patterns support existing trends
• Multiple patterns increase signal strength

<i>Higher confidence = stronger signal</i>"""

        fig.add_annotation(
            x=df.index[8] if len(df) > 8 else df.index[0],
            y=df['High'].max() * 0.996,
            text=legend_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color='white', size=10, family="Arial"),
            bgcolor='rgba(15,15,15,0.95)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1,
            borderpad=12
        )

    def _add_bullish_legend(self, fig: go.Figure, df: pd.DataFrame):
        """Add bullish patterns legend"""
        legend_text = """<b>📈 Bullish Candlestick Patterns</b>

<b>Strong Bullish Signals:</b>
• Bullish Engulfing - Strong reversal
• Morning Star - 3-candle reversal
• Hammer - Support & reversal

<b>Moderate Bullish Signals:</b>
• Inverted Hammer - Potential reversal
• Doji - Indecision (context dependent)

<i>Green markers indicate bullish bias</i>"""

        fig.add_annotation(
            x=df.index[8] if len(df) > 8 else df.index[0],
            y=df['High'].max() * 0.996,
            text=legend_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color='white', size=10, family="Arial"),
            bgcolor='rgba(15,15,15,0.95)',
            bordercolor='rgba(76,175,80,0.3)',
            borderwidth=1,
            borderpad=12
        )

    def _add_bearish_legend(self, fig: go.Figure, df: pd.DataFrame):
        """Add bearish patterns legend"""
        legend_text = """<b>📉 Bearish Candlestick Patterns</b>

<b>Strong Bearish Signals:</b>
• Bearish Engulfing - Strong reversal
• Evening Star - 3-candle reversal
• Shooting Star - Resistance & reversal

<b>Moderate Bearish Signals:</b>
• Hanging Man - Potential reversal
• Doji - Indecision (context dependent)

<i>Red markers indicate bearish bias</i>"""

        fig.add_annotation(
            x=df.index[8] if len(df) > 8 else df.index[0],
            y=df['High'].max() * 0.996,
            text=legend_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color='white', size=10, family="Arial"),
            bgcolor='rgba(15,15,15,0.95)',
            bordercolor='rgba(244,67,54,0.3)',
            borderwidth=1,
            borderpad=12
        )

    def _get_pattern_description(self, pattern_type: PatternType) -> str:
        """Get description for pattern type"""
        descriptions = {
            PatternType.BULLISH_ENGULFING: "Large bullish candle engulfs previous bearish candle",
            PatternType.BEARISH_ENGULFING: "Large bearish candle engulfs previous bullish candle",
            PatternType.HAMMER: "Small body with long lower wick, potential bullish reversal",
            PatternType.HANGING_MAN: "Small body with long lower wick, potential bearish reversal in uptrend",
            PatternType.INVERTED_HAMMER: "Small body with long upper wick, potential bullish reversal",
            PatternType.SHOOTING_STAR: "Small body with long upper wick, potential bearish reversal in uptrend",
            PatternType.DOJI: "Very small body, indecision and potential reversal",
            PatternType.SPINNING_TOP: "Small body with upper and lower wicks, indecision pattern",
            PatternType.MORNING_STAR: "Three-candle bullish reversal pattern",
            PatternType.EVENING_STAR: "Three-candle bearish reversal pattern",
            PatternType.THREE_WHITE_SOLDIERS: "Three consecutive bullish candles, strong uptrend",
            PatternType.THREE_BLACK_CROWS: "Three consecutive bearish candles, strong downtrend",
            PatternType.TWEEZER_TOP: "Two candles with similar highs, resistance rejection",
            PatternType.TWEEZER_BOTTOM: "Two candles with similar lows, support rejection"
        }
        return descriptions.get(pattern_type, "Pattern description not available")

    def _get_pattern_signals(self, pattern_type: PatternType) -> str:
        """Get trading signals for pattern type"""
        signals = {
            PatternType.BULLISH_ENGULFING: "Strong buy signal, potential trend reversal",
            PatternType.BEARISH_ENGULFING: "Strong sell signal, potential trend reversal",
            PatternType.HAMMER: "Buy signal after downtrend, ideal for BOS/CHOCH entries",
            PatternType.HANGING_MAN: "Sell signal in uptrend, wait for confirmation",
            PatternType.INVERTED_HAMMER: "Potential buy signal, needs confirmation",
            PatternType.SHOOTING_STAR: "Sell signal in uptrend, ideal for BOS/CHOCH entries",
            PatternType.DOJI: "Entry warning - indecision, especially near levels",
            PatternType.SPINNING_TOP: "Entry warning - needs confirmation on 15M",
            PatternType.MORNING_STAR: "Strong buy signal, high confluence setup",
            PatternType.EVENING_STAR: "Strong sell signal, high confluence setup",
            PatternType.THREE_WHITE_SOLDIERS: "Strong trend confirmation, continuation signal",
            PatternType.THREE_BLACK_CROWS: "Strong trend confirmation, continuation signal",
            PatternType.TWEEZER_TOP: "Strong resistance rejection, reversal signal",
            PatternType.TWEEZER_BOTTOM: "Strong support rejection, reversal signal"
        }
        return signals.get(pattern_type, "Trading signal not available")
    
    def plot_entry_levels(self, df: pd.DataFrame, patterns: List, entry_levels: List, symbol: str) -> go.Figure:
        """Create chart showing entry levels based on patterns"""
        fig = self._create_base_candlestick_chart(df, f"{symbol} - Entry Levels (95%+ Confidence)", symbol)
        
        # Add entry levels - only show key levels to avoid clutter
        entry_prices = []
        stop_losses = []
        take_profits = []
        
        for entry in entry_levels:
            entry_prices.append(entry['entry_price'])
            stop_losses.append(entry['stop_loss'])
            take_profits.append(entry['take_profit'])
        
        # Show only unique levels to reduce visual clutter
        unique_entries = list(set([round(p, 2) for p in entry_prices]))
        unique_sls = list(set([round(p, 2) for p in stop_losses]))
        unique_tps = list(set([round(p, 2) for p in take_profits]))
        
        # Add key entry levels (max 10 each)
        for entry_price in sorted(unique_entries)[:10]:
            fig.add_hline(
                y=entry_price,
                line_dash="solid",
                line_color='blue',
                line_width=1,
                opacity=0.3,
                annotation_text=f"Entry: ${entry_price:.2f}",
                annotation_position="top right"
            )
        
        # Add key stop loss levels (max 10)
        for sl_price in sorted(unique_sls)[:10]:
            fig.add_hline(
                y=sl_price,
                line_dash="dash",
                line_color='red',
                line_width=1,
                opacity=0.2,
                annotation_text=f"SL: ${sl_price:.2f}",
                annotation_position="bottom right"
            )
        
        # Add key take profit levels (max 10)
        for tp_price in sorted(unique_tps)[:10]:
            fig.add_hline(
                y=tp_price,
                line_dash="dash",
                line_color='green',
                line_width=1,
                opacity=0.2,
                annotation_text=f"TP: ${tp_price:.2f}",
                annotation_position="top right"
            )
        
        # Add pattern markers for each entry level
        for entry in entry_levels:
            timestamp = entry['timestamp']
            pattern_type = entry['pattern_type']
            direction = entry['direction']
            
            # Get color based on direction
            color = '#00E676' if direction == 'Bullish' else '#FF1744' if direction == 'Bearish' else '#9C27B0'
            
            # Add pattern marker
            fig.add_trace(go.Scatter(
                x=[timestamp],
                y=[entry['entry_price']],
                mode='markers',
                marker=dict(
                    symbol=self.pattern_symbols.get(pattern_type, 'circle'),
                    size=12,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                name=f"{pattern_type.value}",
                hovertemplate=f"""
                <b>{pattern_type.value}</b><br>
                Direction: {direction}<br>
                Confidence: {entry['confidence']:.1%}<br>
                Entry: ${entry['entry_price']:.2f}<br>
                Stop Loss: ${entry['stop_loss']:.2f}<br>
                Take Profit: ${entry['take_profit']:.2f}<br>
                Risk:Reward: 1:{entry['risk_reward']:.1f}<br>
                <extra></extra>""",
                showlegend=True
            ))
        
        # Add entry levels legend
        self._add_entry_levels_legend(fig, df, entry_levels)
        
        # Update layout
        fig.update_layout(
            title=f"<b>{symbol} Entry Levels - High Confidence Patterns (95%+)</b>",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.3)",
                borderwidth=1
            )
        )
        
        return fig
    
    def _add_entry_levels_legend(self, fig: go.Figure, df: pd.DataFrame, entry_levels: List):
        """Add entry levels legend"""
        bullish_count = sum(1 for e in entry_levels if e['direction'] == 'Bullish')
        bearish_count = sum(1 for e in entry_levels if e['direction'] == 'Bearish')
        neutral_count = sum(1 for e in entry_levels if e['direction'] == 'Neutral')
        avg_rr = sum(e['risk_reward'] for e in entry_levels) / len(entry_levels) if entry_levels else 0
        
        legend_text = f"""<b>📍 Entry Levels Summary</b>

<b>Pattern Distribution:</b>
• Bullish Entries: {bullish_count}
• Bearish Entries: {bearish_count}
• Neutral Entries: {neutral_count}

<b>Risk Management:</b>
• Average R:R Ratio: 1:{avg_rr:.1f}
• All patterns: 95%+ confidence
• Entry: Solid line
• Stop Loss: Red dashed
• Take Profit: Green dashed

<b>Entry Strategy:</b>
• Bullish: Enter above pattern high
• Bearish: Enter below pattern low
• Wait for confirmation breakout</b>"""

        fig.add_annotation(
            x=df.index[8] if len(df) > 8 else df.index[0],
            y=df['High'].max() * 0.996,
            text=legend_text,
            showarrow=False,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(0,0,0,0.8)',
            font=dict(color='white', size=10),
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1,
            borderpad=12
        )