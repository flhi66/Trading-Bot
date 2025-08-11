import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional

class EnhancedChartPlotter:
    def __init__(self):
        self.color_scheme = {
            ('BOS', 'Bullish'): '#26a69a', ('BOS', 'Bearish'): '#ef5350',
            ('CHOCH', 'Bullish'): '#2196F3', ('CHOCH', 'Bearish'): '#FFA726',
            ('QML', 'Bullish'): '#9C27B0', ('QML', 'Bearish'): '#FF5722',
            ('A+', 'Bullish'): '#4CAF50', ('A+', 'Bearish'): '#F44336'
        }
        
        # Level importance weights for prioritization
        self.level_importance = {
            'BOS': 5,
            'CHOCH': 4,
            'QML': 3,
            'A+': 2,
            'Entry': 1
        }

    def _create_base_chart(self, df: pd.DataFrame, title: str, symbol: str) -> go.Figure:
        """Creates the base candlestick figure."""
        fig = go.Figure(data=go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name=symbol, increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ))
        fig.update_layout(
            title=title, template="plotly_dark", height=900,
            xaxis_rangeslider_visible=False, showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
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
                dash='dot'
            ),
            name=f"{event.event_type.value} Break Line",
            hoverinfo='skip',
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

    def _detect_qml_levels(self, df: pd.DataFrame, events: List) -> List[Dict]:
        """
        Detect Quasimodo (QML) levels based on failed breakouts and rejections.
        QML levels are areas where price attempted to break but failed and reversed strongly.
        """
        qml_levels = []
        
        for i, event in enumerate(events):
            # Look for failed breakouts that create QML levels
            if hasattr(event, 'context') and event.context:
                broken_level = event.broken_level
                
                # Check if this level was tested multiple times before breaking
                test_count = event.context.get('level_test_count', 1)
                rejection_strength = event.context.get('rejection_strength', 0)
                
                # QML criteria: Multiple tests + strong rejection + eventual break
                if test_count >= 2 and rejection_strength > 0.6:
                    qml_level = {
                        'price': broken_level['price'],
                        'timestamp': broken_level['timestamp'],
                        'name': f"QML_{broken_level['name']}",
                        'level_type': 'QML',
                        'direction': event.direction,
                        'strength': rejection_strength,
                        'test_count': test_count,
                        'failed_breakout': True
                    }
                    qml_levels.append(qml_level)
        
        return qml_levels

    def _extract_aplus_levels(self, events: List) -> List[Dict]:
        """
        Extract A+ entry levels from events context.
        A+ levels are premium entry points based on optimal risk-reward setups.
        """
        aplus_levels = []
        
        for event in events:
            if hasattr(event, 'context') and event.context:
                # Extract A+ entry from context
                a_plus_entry = event.context.get('a_plus_entry')
                if a_plus_entry:
                    aplus_level = {
                        'price': a_plus_entry['price'],
                        'timestamp': a_plus_entry['timestamp'],
                        'name': a_plus_entry.get('name', f"A+_Entry_{event.direction}"),
                        'level_type': 'A+',
                        'direction': event.direction,
                        'parent_event': event.event_type.value,
                        'confidence': getattr(event, 'confidence', 0.7),
                        'risk_reward': a_plus_entry.get('risk_reward', 'High'),
                        'entry_type': 'BUY' if event.direction == 'Bullish' else 'SELL'
                    }
                    aplus_levels.append(aplus_level)
                
                # Also check for additional A+ levels in context
                additional_entries = event.context.get('additional_entries', [])
                for entry in additional_entries:
                    if entry.get('grade') == 'A+':
                        aplus_level = {
                            'price': entry['price'],
                            'timestamp': entry['timestamp'],
                            'name': entry.get('name', f"A+_Additional_{event.direction}"),
                            'level_type': 'A+',
                            'direction': event.direction,
                            'parent_event': event.event_type.value,
                            'confidence': entry.get('confidence', 0.6),
                            'risk_reward': entry.get('risk_reward', 'Medium'),
                            'entry_type': 'BUY' if event.direction == 'Bullish' else 'SELL'
                        }
                        aplus_levels.append(aplus_level)
        
        return aplus_levels

    def plot_event_levels(self, df: pd.DataFrame, events: List, symbol: str) -> go.Figure:
        """
        Creates a professional chart showing only A+ entries and QML levels.
        BOS/CHOCH levels are excluded as they're shown in separate charts.
        """
        fig = self._create_base_chart(df, f"{symbol} - Premium Entry Levels (A+ & QML)", symbol)

        # Extract only A+ and QML level types
        qml_levels = self._detect_qml_levels(df, events)
        aplus_levels = self._extract_aplus_levels(events)
        
        # Organize only A+ and QML levels
        level_categories = {
            'QML_Bullish': [], 'QML_Bearish': [],
            'A+_Bullish': [], 'A+_Bearish': []
        }
        
        # Add QML levels with enhanced professional styling
        for qml in qml_levels:
            category = f"QML_{qml['direction']}"
            level_categories[category].append({
                'price': qml['price'],
                'timestamp': qml['timestamp'],
                'name': qml['name'],
                'color': self.color_scheme.get(('QML', qml['direction']), '#9C27B0'),
                'level_type': 'QML',
                'direction': qml['direction'],
                'confidence': qml['strength'],
                'importance': self.level_importance.get('QML', 3),
                'test_count': qml['test_count'],
                'rejection_strength': qml['strength']
            })
        
        # Add A+ levels with premium styling
        for aplus in aplus_levels:
            category = f"A+_{aplus['direction']}"
            level_categories[category].append({
                'price': aplus['price'],
                'timestamp': aplus['timestamp'],
                'name': aplus['name'],
                'color': self.color_scheme.get(('A+', aplus['direction']), '#4CAF50'),
                'level_type': 'A+',
                'direction': aplus['direction'],
                'confidence': aplus['confidence'],
                'importance': self.level_importance.get('A+', 2),
                'entry_type': aplus['entry_type'],
                'risk_reward': aplus['risk_reward'],
                'parent_event': aplus['parent_event']
            })

        def draw_professional_levels(levels: List, level_type: str):
            """Draw levels with professional, clean styling"""
            if not levels:
                return
                
            # Advanced deduplication for cleaner visualization
            unique_levels = self._deduplicate_levels_advanced(levels, threshold=0.002)
            
            for level in unique_levels:
                price = level['price']
                timestamp = level['timestamp']
                color = level['color']
                confidence = level['confidence']
                
                if level_type == 'QML':
                    # QML Professional Styling
                    line_style = "longdashdot"
                    line_width = 3
                    line_opacity = 0.85
                    zone_opacity = 0.12
                    zone_height = abs(price) * 0.0008
                    
                    # QML marker styling
                    marker_symbol = 'star'
                    marker_size = 16
                    marker_line_width = 3
                    marker_line_color = '#FFD700'  # Gold outline
                    
                elif level_type == 'A+':
                    # A+ Premium Styling
                    line_style = "solid"
                    line_width = 4
                    line_opacity = 0.95
                    zone_opacity = 0.15
                    zone_height = abs(price) * 0.0006
                    
                    # A+ marker styling based on entry type
                    if level.get('entry_type') == 'BUY':
                        marker_symbol = 'triangle-up'
                        color = '#00E676'  # Bright green
                    else:
                        marker_symbol = 'triangle-down'
                        color = '#FF1744'  # Bright red
                    
                    marker_size = 15
                    marker_line_width = 3
                    marker_line_color = 'white'
                
                # Draw professional horizontal level line
                fig.add_hline(
                    y=price,
                    line_dash=line_style,
                    line_color=color,
                    line_width=line_width,
                    opacity=line_opacity
                )
                
                # Add premium confidence zones
                if confidence > 0.6:
                    fig.add_hrect(
                        y0=price - zone_height,
                        y1=price + zone_height,
                        fillcolor=color,
                        opacity=zone_opacity,
                        layer="below",
                        line_width=0
                    )
                
                # Add professional markers
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[price],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=marker_size,
                        color=color,
                        line=dict(width=marker_line_width, color=marker_line_color),
                        opacity=0.95
                    ),
                    name=f"{level_type} {level['direction']}",
                    hovertemplate=self._get_professional_hover_template(level),
                    showlegend=False
                ))

        # Draw only A+ and QML levels with professional styling
        draw_professional_levels(level_categories['QML_Bullish'] + level_categories['QML_Bearish'], 'QML')
        draw_professional_levels(level_categories['A+_Bullish'] + level_categories['A+_Bearish'], 'A+')

        # Add clean professional legend
        self._add_professional_legend(fig, df)
        
        # Add simplified statistics for A+ and QML only
        self._add_aplus_qml_statistics(fig, df, qml_levels, aplus_levels)
        
        # Add clean price labels for key levels only
        self._add_clean_price_labels(fig, df, level_categories)

        return fig

    def plot_all_levels(self, df: pd.DataFrame, events: List, symbol: str) -> go.Figure:
        """
        Creates a chart showing all levels (TJL, QML, A+) with minimal markers and hover effects.
        This is Chart 2 - All Trading Levels.
        """
        fig = self._create_base_chart(df, f"{symbol} - All Trading Levels", symbol)

        # Extract and organize all available levels
        all_levels = []
        
        # Add TJL levels (from broken levels in events)
        for event in events:
            broken_level = event.broken_level
            all_levels.append({
                'price': broken_level['price'],
                'timestamp': broken_level['timestamp'],
                'name': broken_level['name'],
                'color': self.color_scheme.get((event.event_type.value, event.direction), '#888888'),
                'level_type': 'TJL',
                'direction': event.direction,
                'confidence': getattr(event, 'confidence', 0.7),
                'importance': self.level_importance.get('Entry', 4),
                'parent_event': event.event_type.value,
                'symbol': 'circle',
                'size': 12
            })
        
        # Add A+ levels (from context entries)
        for event in events:
            if hasattr(event, 'context') and event.context:
                a_plus_entry = event.context.get('a_plus_entry')
                if a_plus_entry:
                    # Determine level type based on name
                    entry_name = a_plus_entry.get('name', '')
                    
                    if 'QML' in entry_name:
                        level_type = 'QML'
                        symbol = 'star'
                        size = 16
                        color = self.color_scheme.get(('QML', event.direction), '#9C27B0')
                    elif 'TJL' in entry_name:
                        level_type = 'A+'
                        symbol = 'triangle-up' if event.direction == 'Bullish' else 'triangle-down'
                        size = 14
                        color = '#00E676' if event.direction == 'Bullish' else '#FF1744'
                    else:
                        level_type = 'A+'
                        symbol = 'triangle-up' if event.direction == 'Bullish' else 'triangle-down'
                        size = 14
                        color = '#00E676' if event.direction == 'Bullish' else '#FF1744'
                    
                    all_levels.append({
                        'price': a_plus_entry['price'],
                        'timestamp': a_plus_entry['timestamp'],
                        'name': entry_name,
                        'color': color,
                        'level_type': level_type,
                        'direction': event.direction,
                        'confidence': getattr(event, 'confidence', 0.7),
                        'importance': self.level_importance.get(level_type, 3),
                        'parent_event': event.event_type.value,
                        'symbol': symbol,
                        'size': size,
                        'entry_type': 'BUY' if event.direction == 'Bullish' else 'SELL'
                    })

        # Remove duplicates based on price proximity
        unique_levels = self._deduplicate_levels_advanced(all_levels, threshold=0.001)
        
        # Draw minimal markers with hover effects (NO horizontal lines)
        for level in unique_levels:
            price = level['price']
            timestamp = level['timestamp']
            color = level['color']
            confidence = level['confidence']
            
            # Add only markers with hover info
            fig.add_trace(go.Scatter(
                x=[timestamp],
                y=[price],
                mode='markers',
                marker=dict(
                    symbol=level['symbol'],
                    size=level['size'],
                    color=color,
                    line=dict(width=2, color='white'),
                    opacity=0.85
                ),
                name=f"{level['level_type']} {level['direction']}",
                hovertemplate=self._get_level_hover_template(level),
                showlegend=False
            ))

        # Add minimal legend
        self._add_minimal_legend(fig, df)
        
        # Add statistics
        self._add_minimal_statistics(fig, df, all_levels)

        return fig

    def plot_entries_only(self, df: pd.DataFrame, events: List, symbol: str) -> go.Figure:
        """
        Creates a chart showing only entry points (triangles) without horizontal lines.
        This is Chart 3 - Entry Points Only.
        """
        fig = self._create_base_chart(df, f"{symbol} - Entry Points Only", symbol)

        # Extract entry levels
        aplus_levels = self._extract_aplus_levels(events)
        
        # Draw only entry markers (no horizontal lines)
        for aplus in aplus_levels:
            price = aplus['price']
            timestamp = aplus['timestamp']
            confidence = aplus['confidence']
            
            # Determine marker style based on entry type
            if aplus['entry_type'] == 'BUY':
                marker_symbol = 'triangle-up'
                color = '#00E676'  # Bright green
                marker_size = 18
            else:
                marker_symbol = 'triangle-down'
                color = '#FF1744'  # Bright red
                marker_size = 18
            
            # Add entry marker (no horizontal line)
            fig.add_trace(go.Scatter(
                x=[timestamp],
                y=[price],
                mode='markers',
                marker=dict(
                    symbol=marker_symbol,
                    size=marker_size,
                    color=color,
                    line=dict(width=3, color='white'),
                    opacity=0.95
                ),
                name=f"A+ {aplus['entry_type']}",
                hovertemplate=self._get_entry_hover_template(aplus),
                showlegend=False
            ))
            
            # Add entry label
            signal_emoji = "🟢" if aplus['entry_type'] == 'BUY' else "🔴"
            label_text = f"{signal_emoji} A+ {aplus['entry_type']} @ ${price:.2f}"
            
            fig.add_annotation(
                x=timestamp,
                y=price,
                text=label_text,
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                ax=0,
                ay=-30 if aplus['entry_type'] == 'BUY' else 30,
                bgcolor=color,
                font=dict(color='white', size=10, weight="bold"),
                borderpad=4,
                bordercolor='white',
                borderwidth=1
            )

        # Add entry-only legend
        self._add_entry_legend(fig, df)
        
        # Add entry statistics
        self._add_entry_statistics(fig, df, aplus_levels)

        return fig

    def plot_enhanced_levels(self, df: pd.DataFrame, events: List, symbol: str) -> go.Figure:
        """
        Alias for plot_event_levels - provides the same enhanced functionality.
        """
        return self.plot_event_levels(df, events, symbol)

    def _deduplicate_levels_advanced(self, levels: List, threshold: float = 0.002) -> List:
        """Advanced deduplication with tighter control for cleaner visualization"""
        if not levels:
            return []
        
        # Sort by importance and confidence
        sorted_levels = sorted(levels, key=lambda x: (x['importance'], x['confidence']), reverse=True)
        unique_levels = []
        
        for level in sorted_levels:
            too_close = False
            for existing in unique_levels:
                price_diff = abs(level['price'] - existing['price'])
                price_threshold = abs(level['price']) * threshold
                if price_diff < price_threshold:
                    too_close = True
                    break
            
            if not too_close:
                unique_levels.append(level)
        
        return unique_levels[:10]  # Limit to top 10 for clean visualization

    def _deduplicate_levels(self, levels: List) -> List:
        """Backward compatibility method - calls advanced deduplication"""
        return self._deduplicate_levels_advanced(levels)

    def _get_professional_hover_template(self, level: Dict) -> str:
        """Professional hover template with essential information only"""
        template = f"<b>🎯 {level['level_type']} {level['direction']}</b><br>"
        template += f"<b>Price: ${level['price']:.2f}</b><br>"
        template += f"Confidence: {level['confidence']:.0%}<br>"
        
        if level['level_type'] == 'QML':
            template += f"Rejection Tests: {level.get('test_count', 'N/A')}<br>"
            template += f"Strength: {level.get('rejection_strength', 0):.0%}<br>"
        
        if level['level_type'] == 'A+':
            template += f"<b>Signal: {level.get('entry_type', 'ENTRY')}</b><br>"
            template += f"Risk/Reward: {level.get('risk_reward', 'High')}<br>"
        
        template += f"Time: {level['timestamp']}<extra></extra>"
        return template

    def _get_entry_hover_template(self, entry: Dict) -> str:
        """Entry-specific hover template for Chart 3"""
        template = f"<b>🎯 A+ Entry Point</b><br>"
        template += f"<b>Signal: {entry.get('entry_type', 'ENTRY')}</b><br>"
        template += f"<b>Price: ${entry['price']:.2f}</b><br>"
        template += f"Confidence: {entry['confidence']:.0%}<br>"
        template += f"Risk/Reward: {entry.get('risk_reward', 'High')}<br>"
        template += f"Parent Event: {entry.get('parent_event', 'N/A')}<br>"
        template += f"Time: {entry['timestamp']}<extra></extra>"
        return template

    def _get_level_hover_template(self, level: Dict) -> str:
        """Hover template for all levels on Chart 2"""
        template = f"<b>{level['level_type']} Level</b><br>"
        template += f"<b>Name: {level['name']}</b><br>"
        template += f"<b>Price: ${level['price']:.2f}</b><br>"
        template += f"Direction: {level['direction']}<br>"
        template += f"Confidence: {level['confidence']:.0%}<br>"
        template += f"Parent Event: {level.get('parent_event', 'N/A')}<br>"
        if 'entry_type' in level:
            template += f"Entry Type: {level['entry_type']}<br>"
        template += f"Time: {level['timestamp']}<extra></extra>"
        return template

    def _add_minimal_legend(self, fig: go.Figure, df: pd.DataFrame):
        """Minimal legend for Chart 2"""
        legend_text = """<b>📊 All Trading Levels</b>

<b>TJL</b> - Take Level (broken structure) ⭕
<b>QML</b> - Quasimodo (failed breakouts) ⭐
<b>A+</b> - Premium entry opportunities ▲▼

<i>Minimal markers only - hover for details</i>"""

        fig.add_annotation(
            x=df.index[8],
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

    def _add_minimal_statistics(self, fig: go.Figure, df: pd.DataFrame, all_levels: List):
        """Minimal statistics for Chart 2"""
        tjl_count = len([l for l in all_levels if l['level_type'] == 'TJL'])
        qml_count = len([l for l in all_levels if l['level_type'] == 'QML'])
        aplus_count = len([l for l in all_levels if l['level_type'] == 'A+'])
        
        stats_text = f"""<b>Level Summary</b>
TJL Levels: {tjl_count}
QML Levels: {qml_count}
A+ Levels: {aplus_count}
Total: {len(all_levels)}"""

        fig.add_annotation(
            x=df.index[-8],
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

    def _add_comprehensive_legend(self, fig: go.Figure, df: pd.DataFrame):
        """Comprehensive legend for all level types"""
        legend_text = """<b>📊 All Trading Levels</b>

<b>TJL</b> - Take Level (broken structure) ⭕
<b>QML</b> - Quasimodo (failed breakouts) ⭐
<b>A+</b> - Premium entry opportunities ▲▼

<b>Visual Guide:</b>
• Dashed lines = TJL levels
• Dash-dot lines = QML levels  
• Solid lines = A+ entries
• Shaded areas = High-confidence zones"""

        fig.add_annotation(
            x=df.index[8],
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

    def _add_entry_legend(self, fig: go.Figure, df: pd.DataFrame):
        """Legend for entry-only chart"""
        legend_text = """<b>🎯 Entry Points Only</b>

<b>A+ Entries</b>
🟢 BUY signals (triangle-up)
🔴 SELL signals (triangle-down)

<i>Only entry markers</i>"""

        fig.add_annotation(
            x=df.index[-8],
            y=df['High'].max(),
            text=legend_text,
            showarrow=False,
            xanchor="right",
            yanchor="top",
            font=dict(color='black', size=10, family="Arial"),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(200,200,200,0.5)',
            borderwidth=1,
            borderpad=12
        )

    def _add_all_levels_statistics(self, fig: go.Figure, df: pd.DataFrame, 
                                qml_levels: List, aplus_levels: List, events: List):
        """Statistics for all levels chart"""
        total_qml = len(qml_levels)
        total_aplus = len(aplus_levels)
        total_tjl = len(events)  # Each event has a broken level (TJL)
        buy_entries = len([a for a in aplus_levels if a['entry_type'] == 'BUY'])
        sell_entries = len([a for a in aplus_levels if a['entry_type'] == 'SELL'])
        
        # Calculate average confidence
        if aplus_levels:
            avg_confidence = sum(a['confidence'] for a in aplus_levels) / len(aplus_levels)
        else:
            avg_confidence = 0
        
        stats_text = f"""<b>All Levels Summary</b>
TJL Levels: {total_tjl}
QML Reversals: {total_qml}
A+ Entries: {total_aplus} (📈 {buy_entries} BUY | 📉 {sell_entries} SELL)
Avg Confidence: {avg_confidence:.0%}"""

        fig.add_annotation(
            x=df.index[-8],
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

    def _add_entry_statistics(self, fig: go.Figure, df: pd.DataFrame, aplus_levels: List):
        """Statistics for entry-only chart"""
        total_entries = len(aplus_levels)
        buy_entries = len([a for a in aplus_levels if a['entry_type'] == 'BUY'])
        sell_entries = len([a for a in aplus_levels if a['entry_type'] == 'SELL'])
        
        # Calculate average confidence
        if aplus_levels:
            avg_confidence = sum(a['confidence'] for a in aplus_levels) / len(aplus_levels)
        else:
            avg_confidence = 0
        
        stats_text = f"""<b>Entry Points Summary</b>
Total Entries: {total_entries}
📈 BUY Signals: {buy_entries}
📉 SELL Signals: {sell_entries}
Avg Confidence: {avg_confidence:.0%}"""

        fig.add_annotation(
            x=df.index[-8],
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

    def _add_professional_legend(self, fig: go.Figure, df: pd.DataFrame):
        """Clean, professional legend for A+ and QML levels only"""
        legend_text = """<b>🎯 Premium Trading Levels</b>

<b>QML (Quasimodo)</b> - Failed breakout reversals ⭐
<b>A+ Entries</b> - Premium entry opportunities ▲▼

<b>Visual Guide:</b>
• Solid lines = A+ entries (high confidence)
• Dash-dot lines = QML levels (rejection zones)
• Shaded areas = High-confidence zones
• Line thickness = Signal strength"""

        fig.add_annotation(
            x=df.index[8],
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

    def _add_aplus_qml_statistics(self, fig: go.Figure, df: pd.DataFrame, 
                                qml_levels: List, aplus_levels: List):
        """Clean statistics panel for A+ and QML levels only"""
        total_qml = len(qml_levels)
        total_aplus = len(aplus_levels)
        buy_entries = len([a for a in aplus_levels if a['entry_type'] == 'BUY'])
        sell_entries = len([a for a in aplus_levels if a['entry_type'] == 'SELL'])
        
        # Calculate average confidence
        if aplus_levels:
            avg_confidence = sum(a['confidence'] for a in aplus_levels) / len(aplus_levels)
        else:
            avg_confidence = 0
        
        stats_text = f"""<b>Premium Levels Summary</b>
QML Reversals: {total_qml}
A+ Entries: {total_aplus} (📈 {buy_entries} BUY | 📉 {sell_entries} SELL)
Avg Confidence: {avg_confidence:.0%}"""

        fig.add_annotation(
            x=df.index[-8],
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

    def _add_clean_price_labels(self, fig: go.Figure, df: pd.DataFrame, level_categories: Dict):
        """Add clean price labels for only the most important A+ and QML levels"""
        all_levels = []
        
        # Collect only A+ and QML levels
        for category in ['QML_Bullish', 'QML_Bearish', 'A+_Bullish', 'A+_Bearish']:
            if category in level_categories:
                all_levels.extend(level_categories[category])
        
        # Check if we have any levels to display
        if not all_levels:
            return
        
        # Sort by importance and confidence, limit to top 6 for clean appearance
        important_levels = sorted(all_levels, 
                                key=lambda x: (x['importance'], x['confidence']), 
                                reverse=True)[:6]
        
        for i, level in enumerate(important_levels):
            if level['level_type'] == 'A+':
                # A+ entry labels with clear signal indication
                signal_emoji = "🟢" if level.get('entry_type') == 'BUY' else "🔴"
                label_text = f"{signal_emoji} A+ {level.get('entry_type', '')} @ ${level['price']:.2f}"
                label_color = '#00E676' if level.get('entry_type') == 'BUY' else '#FF1744'
            else:
                # QML labels with reversal indication
                label_text = f"⭐ QML @ ${level['price']:.2f}"
                label_color = level['color']
            
            # Position labels alternately to avoid overlap
            x_position = df.index[-2] if i % 2 == 0 else df.index[-4]
            
            fig.add_annotation(
                x=x_position,
                y=level['price'],
                text=f" {label_text} ",
                showarrow=False,
                xanchor="left",
                xshift=20,
                font=dict(color='white', size=9, family="Arial", weight="bold"),
                bgcolor=label_color,
                borderpad=6,
                bordercolor='rgba(255,255,255,0.8)',
                borderwidth=1,
                opacity=0.95
            )

    def _get_marker_config(self, level_type: str, entry_type: Optional[str] = None, color: str = 'white') -> Dict:
        """Get marker configuration based on level type"""
        base_config = {
            'size': 10,
            'color': color,
            'line': dict(width=2, color='white'),
            'opacity': 0.9
        }
        
        if level_type == 'BOS':
            base_config['symbol'] = 'square'
            base_config['size'] = 12
        elif level_type == 'CHOCH':
            base_config['symbol'] = 'diamond'
            base_config['size'] = 11
        elif level_type == 'QML':
            base_config['symbol'] = 'star'
            base_config['size'] = 14
            base_config['line']['color'] = 'yellow'
        elif level_type == 'A+':
            if entry_type == 'BUY':
                base_config['symbol'] = 'triangle-up'
                base_config['color'] = '#00ff88'
            else:
                base_config['symbol'] = 'triangle-down'
                base_config['color'] = '#ff4444'
            base_config['size'] = 13
            base_config['line']['width'] = 3
        
        return base_config

    def _get_hover_template(self, level: Dict) -> str:
        """Get hover template based on level information"""
        base_template = f"<b>{level['level_type']} {level['direction']}</b><br>"
        base_template += f"{level['name']}<br>"
        base_template += f"Price: {level['price']:.2f}<br>"
        base_template += f"Confidence: {level['confidence']:.1%}<br>"
        base_template += f"Importance: {level['importance']}/5<br>"
        
        if level['level_type'] == 'QML' and 'test_count' in level:
            base_template += f"Tests: {level['test_count']}<br>"
        
        if level['level_type'] == 'A+' and 'risk_reward' in level:
            base_template += f"R:R: {level['risk_reward']}<br>"
            base_template += f"<b>🎯 {level.get('entry_type', 'ENTRY')}</b><br>"
        
        base_template += f"Time: {level['timestamp']}<extra></extra>"
        return base_template

    def _add_enhanced_legend(self, fig: go.Figure, df: pd.DataFrame, level_categories: Dict):
        """Add comprehensive legend explaining all level types"""
        legend_text = """<b>📊 Enhanced Smart Money Levels</b>
━━━ BOS (Break of Structure) - Major trend shifts
┅┅┅ CHOCH (Change of Character) - Minor trend changes  
━━━ QML (Quasimodo) - Failed breakout reversals
━━━ A+ Entries - Premium entry opportunities

<b>Markers:</b> ⬛ BOS  ◆ CHOCH  ⭐ QML  ▲ BUY A+  ▼ SELL A+

<b>Confidence:</b> Line thickness = Higher confidence
<b>Zones:</b> Shaded areas = High-confidence levels"""

        fig.add_annotation(
            x=df.index[5],
            y=df['High'].max() * 0.997,
            text=legend_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color='white', size=9, family="Arial"),
            bgcolor='rgba(0,0,0,0.9)',
            bordercolor='rgba(255,255,255,0.6)',
            borderwidth=1,
            borderpad=10
        )

    def _add_level_statistics(self, fig: go.Figure, df: pd.DataFrame, level_categories: Dict, 
                            qml_levels: List, aplus_levels: List):
        """Add statistics about detected levels"""
        total_bos = len(level_categories['BOS_Bullish']) + len(level_categories['BOS_Bearish'])
        total_choch = len(level_categories['CHOCH_Bullish']) + len(level_categories['CHOCH_Bearish'])
        total_qml = len(qml_levels)
        total_aplus = len(aplus_levels)
        buy_entries = len([a for a in aplus_levels if a['entry_type'] == 'BUY'])
        sell_entries = len([a for a in aplus_levels if a['entry_type'] == 'SELL'])
        
        stats_text = f"""<b>Level Detection Summary</b>
Structure: BOS {total_bos} | CHOCH {total_choch} | QML {total_qml}
A+ Entries: 📈 {buy_entries} BUY | 📉 {sell_entries} SELL
Total Levels: {total_bos + total_choch + total_qml + total_aplus}"""

        fig.add_annotation(
            x=df.index[-5],
            y=df['Low'].min() * 1.003,
            text=stats_text,
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color='white', size=9, family="monospace"),
            bgcolor='rgba(0,0,0,0.85)',
            bordercolor='rgba(255,255,255,0.5)',
            borderwidth=1,
            borderpad=8
        )

    def _add_price_labels(self, fig: go.Figure, df: pd.DataFrame, level_categories: Dict):
        """Add price labels for the most important levels"""
        all_levels = []
        
        # Collect all levels
        for category, levels in level_categories.items():
            all_levels.extend(levels)
        
        # Sort by importance and select top levels
        important_levels = sorted(all_levels, key=lambda x: (x['importance'], x['confidence']), reverse=True)[:8]
        
        for level in important_levels:
            # Determine label style based on level type
            if level['level_type'] == 'A+':
                label_text = f"🎯 {level.get('entry_type', 'A+')} @ {level['price']:.2f}"
                label_color = level['color']
            elif level['level_type'] == 'QML':
                label_text = f"⭐ QML @ {level['price']:.2f}"
                label_color = level['color']
            else:
                label_text = f"{level['level_type']} @ {level['price']:.2f}"
                label_color = level['color']
            
            fig.add_annotation(
                x=df.index[-1],
                y=level['price'],
                text=f" {label_text} ",
                showarrow=False,
                xanchor="left",
                xshift=15,
                font=dict(color='white', size=8, family="Arial", weight="bold"),
                bgcolor=label_color,
                borderpad=4,
                bordercolor='rgba(255,255,255,0.9)',
                borderwidth=1,
                opacity=0.95
            )

    def plot_event_detections(self, df: pd.DataFrame, events: List, symbol: str) -> go.Figure:
        """
        Creates a clean chart showing BOS/CHOCH event detections with structure break lines.
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
                marker_y_position = event.price - (abs(event.price) * 0.002)
            else:
                symbol_icon = 'triangle-down'
                arrow_y_offset = 50
                marker_y_position = event.price + (abs(event.price) * 0.002)
            
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
        
        return fig

    def plot_combined_analysis(self, df: pd.DataFrame, events: List, symbol: str) -> go.Figure:
        """
        Creates a comprehensive chart combining event detections, levels, QML, and A+ entries.
        """
        fig = self._create_base_chart(df, f"{symbol} - Complete Enhanced SMC Analysis", symbol)
        
        # Get all level types
        qml_levels = self._detect_qml_levels(df, events)
        aplus_levels = self._extract_aplus_levels(events)
        
        # Add event detections
        for event in events:
            color = self.color_scheme.get((event.event_type.value, event.direction), 'grey')
            
            # Add structure break line
            self._add_structure_break_line(fig, event, df)
            
            # Add event markers
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
            
            # Add structure level lines
            fig.add_hline(y=event.broken_level['price'], line_dash="dot", line_color=color, line_width=2, opacity=0.7)
        
        # Add QML levels
        for qml in qml_levels:
            color = self.color_scheme.get(('QML', qml['direction']), '#9C27B0')
            fig.add_hline(y=qml['price'], line_dash="longdash", line_color=color, line_width=2, opacity=0.8)
            
            fig.add_trace(go.Scatter(
                x=[qml['timestamp']],
                y=[qml['price']],
                mode='markers',
                marker=dict(symbol='star', size=12, color=color, line=dict(width=2, color='yellow')),
                name=f"QML {qml['direction']}",
                hovertemplate=f"<b>QML {qml['direction']}</b><br>Price: {qml['price']:.2f}<br>Tests: {qml['test_count']}<extra></extra>",
                showlegend=False
            ))
        
        # Add A+ levels
        for aplus in aplus_levels:
            color = self.color_scheme.get(('A+', aplus['direction']), '#4CAF50')
            fig.add_hline(y=aplus['price'], line_dash="solid", line_color=color, line_width=3, opacity=0.9)
            
            symbol_icon = 'triangle-up' if aplus['entry_type'] == 'BUY' else 'triangle-down'
            fig.add_trace(go.Scatter(
                x=[aplus['timestamp']],
                y=[aplus['price']],
                mode='markers',
                marker=dict(symbol=symbol_icon, size=13, color=color, line=dict(width=3, color='white')),
                name=f"A+ {aplus['entry_type']}",
                hovertemplate=f"<b>A+ {aplus['entry_type']}</b><br>Price: {aplus['price']:.2f}<br>R:R: {aplus['risk_reward']}<extra></extra>",
                showlegend=False
            ))
        
        return fig