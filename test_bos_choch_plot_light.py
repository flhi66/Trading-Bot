#!/usr/bin/env python3
"""
Light Theme BOS/CHOCH Plotting Script
Creates BOS/CHOCH analysis plots with light/white theme for better readability
"""

from core.data_loader import load_and_resample
from core.structure_builder import get_market_analysis
from core.smart_money_concepts import MarketStructureAnalyzer
from utils.level_plotter import EnhancedChartPlotter
import pandas as pd
import webbrowser
import os
from typing import Optional, Dict

class LightThemeChartPlotter(EnhancedChartPlotter):
    """Enhanced chart plotter with light theme styling"""
    
    def __init__(self):
        super().__init__()
        # Override color scheme for light theme
        self.color_scheme = {
            ('BOS', 'Bullish'): '#2E7D32', ('BOS', 'Bearish'): '#D32F2F',
            ('CHOCH', 'Bullish'): '#1976D2', ('CHOCH', 'Bearish'): '#F57C00',
            ('QML', 'Bullish'): '#7B1FA2', ('QML', 'Bearish'): '#E64A19',
            ('A+', 'Bullish'): '#388E3C', ('A+', 'Bearish'): '#D32F2F'
        }
        
        # Light theme specific colors
        self.light_theme_colors = {
            'background': '#FFFFFF',
            'grid': '#E0E0E0',
            'text': '#212121',
            'candlestick_up': '#4CAF50',
            'candlestick_down': '#F44336',
            'border': '#BDBDBD'
        }

    def _create_base_chart(self, df: pd.DataFrame, title: str, symbol: str):
        """Creates the base candlestick figure with light theme."""
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Candlestick(
            x=df.index, 
            open=df['Open'], 
            high=df['High'], 
            low=df['Low'], 
            close=df['Close'],
            name=symbol, 
            increasing_line_color=self.light_theme_colors['candlestick_up'], 
            decreasing_line_color=self.light_theme_colors['candlestick_down']
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color=self.light_theme_colors['text'])
            ),
            template="plotly_white",  # Light theme template
            height=900,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            plot_bgcolor=self.light_theme_colors['background'],
            paper_bgcolor=self.light_theme_colors['background'],
            font=dict(color=self.light_theme_colors['text']),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=self.light_theme_colors['border']
            )
        )
        
        # Update axes styling for light theme
        fig.update_xaxes(
            gridcolor=self.light_theme_colors['grid'],
            zerolinecolor=self.light_theme_colors['border'],
            linecolor=self.light_theme_colors['border'],
            tickfont=dict(color=self.light_theme_colors['text'])
        )
        
        fig.update_yaxes(
            gridcolor=self.light_theme_colors['grid'],
            zerolinecolor=self.light_theme_colors['border'],
            linecolor=self.light_theme_colors['border'],
            tickfont=dict(color=self.light_theme_colors['text'])
        )
        
        return fig

    def _add_structure_break_line(self, fig, event, df: pd.DataFrame):
        """Add dotted line showing structure break with light theme styling"""
        import plotly.graph_objects as go
        
        broken_level = event.broken_level
        color = self.color_scheme.get((event.event_type.value, event.direction), '#757575')
        
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

    def _get_marker_config(self, level_type: str, entry_type: Optional[str] = None, color: str = 'black') -> Dict:
        """Get marker configuration with light theme colors"""
        base_config = super()._get_marker_config(level_type, entry_type, color)
        
        # Override colors for light theme
        if level_type == 'BOS':
            if entry_type == 'Bullish':
                base_config['color'] = '#2E7D32'
            else:
                base_config['color'] = '#D32F2F'
        elif level_type == 'CHOCH':
            if entry_type == 'Bullish':
                base_config['color'] = '#1976D2'
            else:
                base_config['color'] = '#F57C00'
        
        # Ensure line color contrasts with light background
        if 'line' in base_config:
            base_config['line']['color'] = base_config['color']
        
        return base_config

def main():
    print("🔍 LIGHT THEME BOS/CHOCH DETECTION ANALYSIS")
    print("=" * 60)
    
    # === Step 1: Load data ===
    symbol = "XAUUSD_H1.csv"
    # Use 365 days of data instead of 60 for better structure analysis
    resampled = load_and_resample(f"data/{symbol}", days_back=60)
    h1_data = resampled.get("1H")
    
    if h1_data is None or h1_data.empty:
        print(f"❌ ERROR: No data loaded for the '1H' timeframe.")
        return
    
    print(f"📊 Loaded {len(h1_data)} H1 candles from {h1_data.index.min()} to {h1_data.index.max()}")
    
    # === Step 2: Get base market structure ===
    # Use prominence_factor=2.5 for BOS/CHOCH detection (more sensitive than 7.5)
    analysis = get_market_analysis(h1_data, prominence_factor=2.5)
    structure = analysis['structure']
    
    print(f"📊 Total structure points: {len(structure)}")
    
    # Debug: Show structure points around the May 25 - June 1 period (2024)
    print("\n🔍 Debug: Structure points around May 25 - June 1, 2024:")
    target_start = pd.Timestamp('2024-05-25', tz='UTC')
    target_end = pd.Timestamp('2024-06-01', tz='UTC')
    
    relevant_points = []
    for i, point in enumerate(structure):
        timestamp = pd.Timestamp(point['timestamp'])
        if target_start <= timestamp <= target_end:
            relevant_points.append((i, point))
    
    if relevant_points:
        print(f"  Found {len(relevant_points)} relevant points:")
        for idx, point in relevant_points[:10]:  # Show first 10 relevant points
            print(f"    Index {idx}: {point['type']} @ {point['price']:.2f} - {point['timestamp']}")
    else:
        print("  No structure points found in this period")
    
    # Debug: Show recent structure points for context
    print(f"\n🔍 Debug: Recent structure points (last 20):")
    for i, point in enumerate(structure[-20:]):
        print(f"  Index {len(structure)-20+i}: {point['type']} @ {point['price']:.2f} - {point['timestamp']}")
    
    # Debug: Check if we have enough structure points for BOS detection
    if len(structure) < 4:
        print(f"\n⚠️  WARNING: Only {len(structure)} structure points found. BOS/CHOCH detection requires at least 4 points.")
        print("   This might be due to the prominence_factor being too restrictive.")
        print("   Consider using more data or adjusting the prominence_factor in structure_builder.py")
    else:
        print(f"\n✅ Sufficient structure points ({len(structure)}) for BOS/CHOCH detection.")
    
    # Debug: Check trend states around recent key points
    if len(structure) >= 10:
        print(f"\n🔍 Debug: Trend states around recent key points:")
        analyzer_debug = MarketStructureAnalyzer(confidence_threshold=0.1)
        
        # Convert structure to the format expected by the analyzer
        structure_objects = []
        for i, point in enumerate(structure):
            from core.smart_money_concepts import StructurePoint, SwingType
            structure_objects.append(StructurePoint(
                timestamp=pd.Timestamp(point['timestamp']),
                price=point['price'],
                swing_type=SwingType(point['type'])
            ))
        
        # Check trend states around recent indices
        for i in range(max(0, len(structure_objects)-10), len(structure_objects)):
            trend_state = analyzer_debug._get_trend_state(structure_objects, i)
            point = structure_objects[i]
            print(f"  Index {i}: {point.swing_type.value} @ {point.price:.2f} - Trend: {trend_state}")

    # === Step 3: Use the analyzer to detect events ===
    analyzer = MarketStructureAnalyzer(confidence_threshold=0.5)
    all_events = analyzer.get_market_events(structure)
    print(f"\nFound {len(all_events)} total events with confidence >= {analyzer.confidence_threshold}")
    
    # Debug: Check what confidence levels are available
    if len(all_events) == 0:
        print("🔍 Debug: No events found with 50% confidence. Checking lower thresholds...")
        # Try with lower threshold to see what's available
        analyzer_lower = MarketStructureAnalyzer(confidence_threshold=0.1)
        lower_events = analyzer_lower.get_market_events(structure)
        if len(lower_events) > 0:
            confidences = [event.confidence for event in lower_events]
            print(f"🔍 Available confidence levels: {sorted(confidences, reverse=True)[:10]}")
            print(f"🔍 Highest confidence: {max(confidences):.3f}")
            print(f"🔍 Events with 50%+ confidence: {sum(1 for c in confidences if c >= 0.5)}")
            print(f"🔍 Events with 70%+ confidence: {sum(1 for c in confidences if c >= 0.7)}")
        else:
            print("🔍 No events found even with 10% confidence threshold")
    
    # Debug: Show all events with their details
    if all_events:
        print(f"\n🔍 Debug: All detected events:")
        for i, event in enumerate(all_events):
            print(f"  Event {i+1}: {event.event_type.value} - {event.direction} @ {event.price:.2f}")
            print(f"    Confidence: {event.confidence:.3f}")
            print(f"    Time: {event.timestamp}")
            print(f"    Description: {event.description}")
            print()
    
    # === Step 4: Generate and show three plots with light theme ===
    if all_events:
        print("\n✅ Generating light theme plots...")
        plotter = LightThemeChartPlotter()
        
        # Debug: Check what levels are being detected
        qml_levels = plotter._detect_qml_levels(h1_data, all_events)
        aplus_levels = plotter._extract_aplus_levels(all_events)
        
        # Count different level types from Chart 2
        chart2_levels = []
        for event in all_events:
            # TJL levels
            chart2_levels.append(('TJL', event.broken_level['name'], event.broken_level['price']))
            
            # A+ levels from context
            if hasattr(event, 'context') and event.context:
                a_plus_entry = event.context.get('a_plus_entry')
                if a_plus_entry:
                    entry_name = a_plus_entry.get('name', '')
                    level_type = 'QML' if 'QML' in entry_name else 'A+'
                    chart2_levels.append((level_type, entry_name, a_plus_entry['price']))
        
        print(f"🔍 Debug: Chart 2 will show {len(chart2_levels)} total levels:")
        
        # Group by type
        from collections import Counter
        type_counts = Counter([level[0] for level in chart2_levels])
        for level_type, count in type_counts.items():
            print(f"  - {level_type}: {count} levels")
        
        print(f"🔍 Debug: Chart 3 will show {len(aplus_levels)} A+ entry points")
        
        # Create output directory for light theme plots
        output_dir = "generated_light_theme_plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # --- Plot 1: Show only the event detections with light theme ---
        fig_detections = plotter.plot_event_detections(
            df=h1_data,
            events=all_events,
            symbol=symbol.split('_')[0]
        )
        detections_file = os.path.join(output_dir, "light_theme_event_detections.html")
        fig_detections.write_html(detections_file)
        print(f"-> Light Theme Chart 1 saved: {detections_file}")
        webbrowser.open(f"file://{os.path.abspath(detections_file)}")

        # --- Plot 2: Show all levels (TJL, QML, A+) with horizontal lines and light theme ---
        fig_all_levels = plotter.plot_all_levels(
            df=h1_data,
            events=all_events,
            symbol=symbol.split('_')[0]
        )
        all_levels_file = os.path.join(output_dir, "light_theme_all_trading_levels.html")
        fig_all_levels.write_html(all_levels_file)
        print(f"-> Light Theme Chart 2 saved: {all_levels_file}")
        webbrowser.open(f"file://{os.path.abspath(all_levels_file)}")

        # --- Plot 3: Show only entries (triangles) without horizontal lines and light theme ---
        fig_entries_only = plotter.plot_entries_only(
            df=h1_data,
            events=all_events,
            symbol=symbol.split('_')[0]
        )
        entries_file = os.path.join(output_dir, "light_theme_entry_points_only.html")
        fig_entries_only.write_html(entries_file)
        print(f"-> Light Theme Chart 3 saved: {entries_file}")
        webbrowser.open(f"file://{os.path.abspath(entries_file)}")
        
        # --- Plot 4: Enhanced levels with light theme ---
        fig_enhanced = plotter.plot_enhanced_levels(
            df=h1_data,
            events=all_events,
            symbol=symbol.split('_')[0]
        )
        enhanced_file = os.path.join(output_dir, "light_theme_enhanced_levels.html")
        fig_enhanced.write_html(enhanced_file)
        print(f"-> Light Theme Chart 4 saved: {enhanced_file}")
        webbrowser.open(f"file://{os.path.abspath(enhanced_file)}")
        
        print(f"\n✅ All light theme charts have been saved to the '{output_dir}' folder and opened in your browser!")
        print(f"📁 You can find the HTML files in: {os.path.abspath(output_dir)}")
        print(f"🎨 Theme: Light/White background for better readability")

    else:
        print("\n✅ Analysis complete. No high-confidence market events to plot.")

if __name__ == "__main__":
    main()
