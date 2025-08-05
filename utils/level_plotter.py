import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict

class EnhancedChartPlotter:
    def __init__(self):
        self.color_scheme = {
            ('BOS', 'Bullish'): '#26a69a', ('BOS', 'Bearish'): '#ef5350',
            ('CHOCH', 'Bullish'): '#2196F3', ('CHOCH', 'Bearish'): '#FFA726'
        }

    def _create_base_chart(self, df: pd.DataFrame, title: str, symbol: str) -> go.Figure:
        """Creates the base candlestick figure."""
        fig = go.Figure(data=go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name=symbol, increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ))
        fig.update_layout(
            title=title, template="plotly_dark", height=800,
            xaxis_rangeslider_visible=False, showlegend=False
        )
        return fig

    def _add_structure_break_line(self, fig: go.Figure, event, df: pd.DataFrame):
        """Add dotted line showing structure break from broken level to break point"""
        broken_level = event.broken_level
        color = self.color_scheme.get((event.event_type.value, event.direction), 'grey')
        
        # Create dotted line from broken level to break point
        fig.add_trace(go.Scatter(
            x=[broken_level['timestamp'], event.timestamp],
            y=[broken_level['price'], event.price],
            mode='lines',
            line=dict(
                color=color,
                width=2,
                dash='dot'  # Dotted line
            ),
            name=f"{event.event_type.value} Break Line",
            hoverinfo='skip',  # Don't show hover info for break lines
            showlegend=False
        ))
        
        # Add a small circle at the broken level point
        fig.add_trace(go.Scatter(
            x=[broken_level['timestamp']],
            y=[broken_level['price']],
            mode='markers',
            marker=dict(
                symbol='circle-open',
                size=8,
                color=color,
                line=dict(width=2, color=color)
            ),
            name=f"Broken {broken_level['name']}",
            hoverinfo='skip',
            showlegend=False
        ))

    def plot_event_detections(self, df: pd.DataFrame, events: List, symbol: str) -> go.Figure:
        """
        Creates a clean chart showing BOS/CHOCH event detections with structure break lines.
        REMOVED: Background zones for cleaner visualization.
        """
        fig = self._create_base_chart(df, f"{symbol} - BOS/CHOCH Event Detections", symbol)

        for event in events:
            color = self.color_scheme.get((event.event_type.value, event.direction), 'grey')
            
            # Add structure break line
            self._add_structure_break_line(fig, event, df)
            
            # Determine symbol and position for event marker
            if event.direction == 'Bullish':
                symbol_icon = 'triangle-up'
                arrow_y_offset = -50
                marker_y_position = event.price - (abs(event.price) * 0.002)  # Slightly below
            else:
                symbol_icon = 'triangle-down'
                arrow_y_offset = 50
                marker_y_position = event.price + (abs(event.price) * 0.002)  # Slightly above
            
            # Use diamond for CHOCH events
            if event.event_type.value == 'CHOCH':
                symbol_icon = 'diamond'
                marker_y_position = event.price
            
            # Add event marker
            fig.add_trace(go.Scatter(
                x=[event.timestamp], 
                y=[marker_y_position], 
                mode='markers',
                marker=dict(
                    symbol=symbol_icon, 
                    size=12, 
                    color=color, 
                    line=dict(width=2, color='white')
                ),
                name=f"{event.event_type.value} {event.direction}",
                hovertemplate=f"<b>{event.event_type.value}</b><br>" +
                             f"Direction: {event.direction}<br>" +
                             f"Price: {event.price:.2f}<br>" +
                             f"Confidence: {event.confidence:.1%}<br>" +
                             f"Time: {event.timestamp}<br>" +
                             f"<i>{event.description}</i><extra></extra>",
                showlegend=False
            ))
            
            # Add clean event label
            event_label = f"<b>{event.event_type.value}</b><br>@{event.price:.2f}"
            fig.add_annotation(
                x=event.timestamp, 
                y=marker_y_position,
                text=event_label, 
                showarrow=True, 
                arrowhead=2, 
                arrowcolor=color,
                ax=0, 
                ay=arrow_y_offset,
                bgcolor=color, 
                font=dict(color='white', size=10), 
                borderpad=4,
                bordercolor='white',
                borderwidth=1
            )
            
            # Add level labels for key levels (simplified)
            broken_level = event.broken_level
            fig.add_annotation(
                x=broken_level['timestamp'],
                y=broken_level['price'],
                text=f"{broken_level['name']}",
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(color=color, size=8, family="Arial Black"),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor=color,
                borderwidth=1,
                borderpad=2
            )
        
        return fig

    def plot_event_levels(self, df: pd.DataFrame, events: List, symbol: str) -> go.Figure:
        """
        Creates a ultra-clean and organized chart showing key trading levels with proper buy/sell entry indications.
        IMPROVED: Better level detection logic, cleaner visualization, proper entry signals.
        """
        fig = self._create_base_chart(df, f"{symbol} - Smart Money Concept Levels", symbol)

        # Enhanced level categorization with better logic
        level_categories = {
            'BOS_Bullish': [], 'BOS_Bearish': [], 
            'CHOCH_Bullish': [], 'CHOCH_Bearish': [], 
            'Entry_Buy': [], 'Entry_Sell': []
        }
        
        # Process events with improved logic
        for event in events:
            event_color = self.color_scheme.get((event.event_type.value, event.direction), 'grey')
            
            # Categorize main structure levels
            category = f"{event.event_type.value}_{event.direction}"
            if category in level_categories:
                level_categories[category].append({
                    'price': event.broken_level['price'],
                    'timestamp': event.broken_level['timestamp'],
                    'name': event.broken_level['name'],
                    'color': event_color,
                    'event_type': event.event_type.value,
                    'direction': event.direction,
                    'confidence': getattr(event, 'confidence', 0.5)
                })
            
            # Process entry levels with proper buy/sell categorization
            a_plus_entry = event.context.get('a_plus_entry')
            if a_plus_entry:
                entry_category = 'Entry_Buy' if event.direction == 'Bullish' else 'Entry_Sell'
                level_categories[entry_category].append({
                    'price': a_plus_entry['price'],
                    'timestamp': a_plus_entry['timestamp'],
                    'name': a_plus_entry['name'],
                    'color': event_color,
                    'event_type': 'Entry',
                    'direction': event.direction,
                    'confidence': getattr(event, 'confidence', 0.5),
                    'trade_type': 'BUY' if event.direction == 'Bullish' else 'SELL'
                })

        def draw_enhanced_level_group(levels: List, line_style: str, line_width: int, show_zones: bool = False, is_entry: bool = False):
            """Enhanced level drawing with better deduplication and styling"""
            if not levels:
                return
                
            # Advanced deduplication - group by price ranges
            price_groups = {}
            for level in levels:
                price_key = round(level['price'] / 5) * 5  # Group by 5-point ranges
                if price_key not in price_groups:
                    price_groups[price_key] = []
                price_groups[price_key].append(level)
            
            # Keep highest confidence level from each group
            unique_levels = []
            for group in price_groups.values():
                best_level = max(group, key=lambda x: x['confidence'])
                unique_levels.append(best_level)
            
            for level in unique_levels:
                price = level['price']
                timestamp = level['timestamp']
                color = level['color']
                name = level['name']
                event_type = level['event_type']
                direction = level['direction']
                
                # Enhanced line styling based on importance
                line_opacity = 0.9 if event_type in ['BOS', 'CHOCH'] else 0.7
                actual_line_width = line_width + (1 if level['confidence'] > 0.7 else 0)
                
                # Draw horizontal level line
                fig.add_hline(
                    y=price,
                    line_dash=line_style,
                    line_color=color,
                    line_width=actual_line_width,
                    opacity=line_opacity
                )
                
                # Add zones for high-confidence levels only
                if show_zones and level['confidence'] > 0.6:
                    zone_height = abs(price) * 0.0004  # Smaller zones
                    fig.add_hrect(
                        y0=price - zone_height,
                        y1=price + zone_height,
                        fillcolor=color,
                        opacity=0.06,
                        layer="below",
                        line_width=0
                    )
                
                # Enhanced markers with proper entry indication
                if is_entry:
                    # Special markers for entry levels
                    if level.get('trade_type') == 'BUY':
                        marker_symbol = 'triangle-up'
                        marker_color = '#00ff88'  # Bright green for buy
                    else:
                        marker_symbol = 'triangle-down'
                        marker_color = '#ff4444'  # Bright red for sell
                    marker_size = 10
                else:
                    # Standard markers for structure levels
                    marker_symbol = 'diamond' if event_type == 'CHOCH' else 'square'
                    marker_color = color
                    marker_size = 8
                
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[price],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=marker_size,
                        color=marker_color,
                        line=dict(width=2, color='white'),
                        opacity=0.95
                    ),
                    name=f"{event_type} - {name}",
                    hovertemplate=(
                        f"<b>{event_type} {direction}</b><br>"
                        f"{name}<br>"
                        f"Price: {price:.2f}<br>"
                        f"Confidence: {level['confidence']:.1%}<br>"
                        f"Time: {timestamp}"
                        f"<br><b>{level.get('trade_type', '')}</b>" if is_entry else "" +
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))

        # Draw levels with enhanced styling
        draw_enhanced_level_group(level_categories['BOS_Bullish'], "dash", 2, True)
        draw_enhanced_level_group(level_categories['BOS_Bearish'], "dash", 2, True)
        draw_enhanced_level_group(level_categories['CHOCH_Bullish'], "dot", 2, False)
        draw_enhanced_level_group(level_categories['CHOCH_Bearish'], "dot", 2, False)
        draw_enhanced_level_group(level_categories['Entry_Buy'], "dashdot", 2, False, True)
        draw_enhanced_level_group(level_categories['Entry_Sell'], "dashdot", 2, False, True)

        # Smart label management - only show most important levels
        all_significant_levels = []
        
        # Prioritize levels by importance
        importance_weights = {'BOS': 3, 'CHOCH': 2, 'Entry': 1}
        
        for category, levels in level_categories.items():
            for level in levels:
                importance = importance_weights.get(level['event_type'], 0) * level['confidence']
                level['importance'] = importance
                all_significant_levels.append(level)
        
        # Sort by importance and remove overlapping levels
        all_significant_levels.sort(key=lambda x: x['importance'], reverse=True)
        
        final_levels = []
        for level in all_significant_levels[:12]:  # Top 12 most important
            # Check if too close to existing levels
            too_close = False
            for existing in final_levels:
                if abs(level['price'] - existing['price']) < abs(level['price']) * 0.005:  # 0.5% threshold
                    too_close = True
                    break
            
            if not too_close:
                final_levels.append(level)
        
        # Add clean, organized labels
        for level in final_levels:
            # Determine label content based on level type
            if level['event_type'] == 'Entry':
                label_text = f"🎯 {level['trade_type']} @ {level['price']:.2f}"
                label_color = '#00ff88' if level['trade_type'] == 'BUY' else '#ff4444'
            else:
                label_text = f"{level['name']} @ {level['price']:.2f}"
                label_color = level['color']
            
            fig.add_annotation(
                x=df.index[-1],
                y=level['price'],
                text=f" {label_text} ",
                showarrow=False,
                xanchor="left",
                xshift=12,
                font=dict(color='white', size=8, family="Arial", weight="bold"),
                bgcolor=label_color,
                borderpad=3,
                bordercolor='rgba(255,255,255,0.8)',
                borderwidth=1,
                opacity=0.95
            )

        # Enhanced legend with entry signals
        legend_text = """<b>📊 Smart Money Levels</b>
━━ BOS (Break of Structure)    ┅┅ CHOCH (Change of Character)
━┅━ Entry Levels    🎯 Trade Signals

<b>Markers:</b> ⬛ BOS  ◆ CHOCH  ▲ BUY  ▼ SELL"""

        fig.add_annotation(
            x=df.index[3],
            y=df['High'].max() * 0.998,
            text=legend_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color='white', size=9, family="Arial"),
            bgcolor='rgba(0,0,0,0.85)',
            bordercolor='rgba(255,255,255,0.5)',
            borderwidth=1,
            borderpad=8
        )

        # Enhanced statistics with entry signals
        total_bos = len(level_categories['BOS_Bullish']) + len(level_categories['BOS_Bearish'])
        total_choch = len(level_categories['CHOCH_Bullish']) + len(level_categories['CHOCH_Bearish'])
        total_buy_entries = len(level_categories['Entry_Buy'])
        total_sell_entries = len(level_categories['Entry_Sell'])
        
        stats_text = f"""Structure: BOS {total_bos} | CHOCH {total_choch}
Entries: 📈 {total_buy_entries} BUY | 📉 {total_sell_entries} SELL"""

        fig.add_annotation(
            x=df.index[-3],
            y=df['Low'].min() * 1.002,
            text=stats_text,
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color='white', size=8, family="monospace"),
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='rgba(255,255,255,0.4)',
            borderwidth=1,
            borderpad=5
        )

        return fig

    def plot_combined_analysis(self, df: pd.DataFrame, events: List, symbol: str) -> go.Figure:
        """
        Creates a comprehensive chart combining both event detections and levels.
        """
        fig = self._create_base_chart(df, f"{symbol} - Complete SMC Analysis", symbol)
        
        # Add all elements from both plots
        for event in events:
            color = self.color_scheme.get((event.event_type.value, event.direction), 'grey')
            
            # Add structure break line
            self._add_structure_break_line(fig, event, df)
            
            # Add event markers (simplified for combined view)
            symbol_icon = 'diamond' if event.event_type.value == 'CHOCH' else ('triangle-up' if event.direction == 'Bullish' else 'triangle-down')
            
            fig.add_trace(go.Scatter(
                x=[event.timestamp], 
                y=[event.price], 
                mode='markers',
                marker=dict(symbol=symbol_icon, size=10, color=color, line=dict(width=1, color='white')),
                name=f"{event.event_type.value} {event.direction}",
                hovertemplate=f"<b>{event.event_type.value}</b><br>Price: {event.price:.2f}<br>Confidence: {event.confidence:.1%}<extra></extra>",
                showlegend=False
            ))
            
            # Add key level lines
            fig.add_hline(y=event.broken_level['price'], line_dash="dot", line_color=color, line_width=1, opacity=0.6)
            
            a_plus_entry = event.context.get('a_plus_entry')
            if a_plus_entry:
                fig.add_hline(y=a_plus_entry['price'], line_dash="dashdot", line_color=color, line_width=1, opacity=0.4)
        
        return fig