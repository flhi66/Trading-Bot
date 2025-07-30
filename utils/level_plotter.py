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
        Creates a ultra-clean and organized chart showing key trading levels.
        MAJOR IMPROVEMENTS: Reduced visual clutter, better grouping, cleaner labels.
        """
        fig = self._create_base_chart(df, f"{symbol} - Key Trading Levels", symbol)

        # Collect and organize all levels first
        all_levels = {}
        level_categories = {'BOS_Bullish': [], 'BOS_Bearish': [], 'CHOCH_Bullish': [], 'CHOCH_Bearish': [], 'Entry': []}
        
        for event in events:
            # Categorize broken levels
            category = f"{event.event_type.value}_{event.direction}"
            if category in level_categories:
                level_categories[category].append({
                    'price': event.broken_level['price'],
                    'timestamp': event.broken_level['timestamp'],
                    'name': event.broken_level['name'],
                    'color': self.color_scheme.get((event.event_type.value, event.direction), 'grey'),
                    'event_type': event.event_type.value
                })
            
            # Add entry levels
            a_plus_entry = event.context.get('a_plus_entry')
            if a_plus_entry:
                level_categories['Entry'].append({
                    'price': a_plus_entry['price'],
                    'timestamp': a_plus_entry['timestamp'],
                    'name': a_plus_entry['name'],
                    'color': self.color_scheme.get((event.event_type.value, event.direction), 'grey'),
                    'event_type': 'Entry'
                })

        def draw_clean_level_group(levels: List, line_style: str, line_width: int, show_zones: bool = False):
            """Draw a group of levels with consistent styling"""
            # Remove duplicate levels by price (keep the most recent)
            unique_levels = {}
            for level in levels:
                price_key = round(level['price'], 2)
                if price_key not in unique_levels or level['timestamp'] > unique_levels[price_key]['timestamp']:
                    unique_levels[price_key] = level
            
            for level in unique_levels.values():
                price = level['price']
                timestamp = level['timestamp']
                color = level['color']
                name = level['name']
                event_type = level['event_type']
                
                # Draw horizontal level line
                fig.add_hline(
                    y=price,
                    line_dash=line_style,
                    line_color=color,
                    line_width=line_width,
                    opacity=0.85
                )
                
                # Add subtle zone only for major levels
                if show_zones:
                    zone_height = abs(price) * 0.0005
                    fig.add_hrect(
                        y0=price - zone_height,
                        y1=price + zone_height,
                        fillcolor=color,
                        opacity=0.08,
                        layer="below",
                        line_width=0
                    )
                
                # Add single marker at level origin
                marker_symbol = 'square' if event_type in ['BOS', 'CHOCH'] else 'circle'
                if event_type == 'CHOCH':
                    marker_symbol = 'diamond'
                
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[price],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=8,
                        color=color,
                        line=dict(width=1.5, color='white'),
                        opacity=0.9
                    ),
                    name=f"{event_type} - {name}",
                    hovertemplate=f"<b>{event_type}</b><br>{name}<br>Price: {price:.2f}<br>Time: {timestamp}<extra></extra>",
                    showlegend=False
                ))

        # Draw level groups with different styling
        draw_clean_level_group(level_categories['BOS_Bullish'], "dash", 2, True)
        draw_clean_level_group(level_categories['BOS_Bearish'], "dash", 2, True)
        draw_clean_level_group(level_categories['CHOCH_Bullish'], "dot", 2, False)
        draw_clean_level_group(level_categories['CHOCH_Bearish'], "dot", 2, False)
        draw_clean_level_group(level_categories['Entry'], "dashdot", 1, False)

        # Create organized right-side labels (only for significant levels)
        significant_levels = []
        for category, levels in level_categories.items():
            if levels:
                # Group levels by price ranges to avoid overlapping labels
                levels_by_price = sorted(levels, key=lambda x: x['price'])
                for i, level in enumerate(levels_by_price):
                    # Only show labels for levels that aren't too close to others
                    show_label = True
                    for other in levels_by_price[i+1:i+3]:  # Check next 2 levels
                        if abs(level['price'] - other['price']) < abs(level['price']) * 0.003:  # Less than 0.3%
                            show_label = False
                            break
                    
                    if show_label:
                        significant_levels.append(level)

        # Sort significant levels by price for clean right-side labeling
        significant_levels.sort(key=lambda x: x['price'], reverse=True)
        
        # Add clean, non-overlapping labels on the right
        for i, level in enumerate(significant_levels[:15]):  # Limit to top 15 levels
            label_text = f"{level['name']} @ {level['price']:.2f}"
            
            fig.add_annotation(
                x=df.index[-1],
                y=level['price'],
                text=f" {label_text} ",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font=dict(color='white', size=8, family="Arial"),
                bgcolor=level['color'],
                borderpad=3,
                bordercolor='rgba(255,255,255,0.6)',
                borderwidth=1,
                opacity=0.9
            )

        # Minimal, clean legend
        legend_text = """<b>Key Levels</b>
━━ BOS Levels    ┅┅ CHOCH Levels    ━┅━ Entry Levels
⬛ Major Levels    ◆ CHOCH    ● Entry Points"""

        fig.add_annotation(
            x=df.index[5],
            y=df['High'].max() * 0.999,
            text=legend_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color='white', size=9, family="Arial"),
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='rgba(255,255,255,0.4)',
            borderwidth=1,
            borderpad=6
        )

        # Compact statistics
        total_bos = len(level_categories['BOS_Bullish']) + len(level_categories['BOS_Bearish'])
        total_choch = len(level_categories['CHOCH_Bullish']) + len(level_categories['CHOCH_Bearish'])
        total_entry = len(level_categories['Entry'])
        
        stats_text = f"BOS: {total_bos} | CHOCH: {total_choch} | Entry: {total_entry}"

        fig.add_annotation(
            x=df.index[-5],
            y=df['Low'].min() * 1.001,
            text=stats_text,
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color='white', size=8, family="monospace"),
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1,
            borderpad=4
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