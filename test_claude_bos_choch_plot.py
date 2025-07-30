import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List
from claude_bos_choch import MarketEvent, EventType

class EnhancedChartPlotter:
    def __init__(self):
        self.color_scheme = {
            "BOS_BULLISH": "#00FF88",      # Bright green
            "BOS_BEARISH": "#FF4444",      # Red
            "CHOCH_BULLISH": "#00AAFF",    # Blue
            "CHOCH_BEARISH": "#FF8800",    # Orange
            "STRUCTURE_LINE": "#666666",    # Gray
            "BROKEN_LEVEL": "#FFFF00"      # Yellow
        }
        
    def create_enhanced_chart(self, df: pd.DataFrame, events: List[MarketEvent], 
                            structure_points: List = None) -> go.Figure:
        """
        Create an enhanced chart with BOS/CHOCH events and structure lines.
        
        Args:
            df: OHLC data with columns: timestamp, open, high, low, close
            events: List of MarketEvent objects from detection logic
            structure_points: Optional list of swing points
        """
        
        # Create candlestick chart
        fig = go.Figure(data=go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="XAUUSD"
        ))
        
        # Add structure lines if provided
        if structure_points:
            self._add_structure_lines(fig, structure_points)
            
        # Add BOS/CHOCH events
        self._add_events_to_chart(fig, events)
        
        # Add broken levels
        self._add_broken_levels(fig, events)
        
        # Configure layout
        fig.update_layout(
            title="XAUUSD - Enhanced BOS/CHOCH Detection",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Remove range slider for cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        return fig
    
    def _add_structure_lines(self, fig: go.Figure, structure_points: List):
        """Add swing high/low structure lines to the chart."""
        for i in range(len(structure_points) - 1):
            current = structure_points[i]
            next_point = structure_points[i + 1]
            
            fig.add_trace(go.Scatter(
                x=[current['timestamp'], next_point['timestamp']],
                y=[current['price'], next_point['price']],
                mode='lines',
                line=dict(color=self.color_scheme["STRUCTURE_LINE"], width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    def _add_events_to_chart(self, fig: go.Figure, events: List[MarketEvent]):
        """Add BOS/CHOCH event markers to the chart."""
        
        # Group events by type for better visualization
        bos_bullish = [e for e in events if e.event_type == EventType.BOS and e.direction == "Bullish"]
        bos_bearish = [e for e in events if e.event_type == EventType.BOS and e.direction == "Bearish"]
        choch_bullish = [e for e in events if e.event_type == EventType.CHOCH and e.direction == "Bullish"]
        choch_bearish = [e for e in events if e.event_type == EventType.CHOCH and e.direction == "Bearish"]
        
        # Add markers for each event type
        event_groups = [
            (bos_bullish, "BOS ↑", self.color_scheme["BOS_BULLISH"], "triangle-up"),
            (bos_bearish, "BOS ↓", self.color_scheme["BOS_BEARISH"], "triangle-down"),
            (choch_bullish, "CHOCH ↑", self.color_scheme["CHOCH_BULLISH"], "diamond"),
            (choch_bearish, "CHOCH ↓", self.color_scheme["CHOCH_BEARISH"], "diamond")
        ]
        
        for events_group, label, color, symbol in event_groups:
            if events_group:
                timestamps = [e.timestamp for e in events_group]
                prices = [e.price for e in events_group]
                confidences = [e.confidence for e in events_group]
                descriptions = [e.description for e in events_group]
                
                # Create hover text with detailed information
                hover_text = [
                    f"{label}<br>"
                    f"Price: {price:.2f}<br>"
                    f"Confidence: {conf:.2f}<br>"
                    f"Time: {ts}<br>"
                    f"{desc}"
                    for price, conf, ts, desc in zip(prices, confidences, timestamps, descriptions)
                ]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=prices,
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=12,
                        color=color,
                        line=dict(width=2, color='white')
                    ),
                    name=label,
                    hovertext=hover_text,
                    hoverinfo='text'
                ))
    
    def _add_broken_levels(self, fig: go.Figure, events: List[MarketEvent]):
        """Add horizontal lines showing broken levels."""
        for event in events:
            broken_level = event.broken_level
            
            # Add horizontal line for broken level
            fig.add_hline(
                y=broken_level["price"],
                line=dict(
                    color=self.color_scheme["BROKEN_LEVEL"],
                    width=1,
                    dash="dash"
                ),
                annotation_text=f"Broken: {broken_level['price']:.2f}",
                annotation_position="right"
            )
    
    def create_confidence_heatmap(self, events: List[MarketEvent]) -> go.Figure:
        """Create a heatmap showing confidence levels of detected events."""
        if not events:
            return go.Figure()
            
        # Prepare data for heatmap
        timestamps = [e.timestamp for e in events]
        prices = [e.price for e in events]
        confidences = [e.confidence for e in events]
        event_types = [f"{e.event_type.value}_{e.direction}" for e in events]
        
        df_heatmap = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'confidence': confidences,
            'event_type': event_types
        })
        
        # Create scatter plot with confidence as color
        fig = px.scatter(
            df_heatmap, 
            x='timestamp', 
            y='price',
            color='confidence',
            size='confidence',
            hover_data=['event_type'],
            color_continuous_scale='RdYlGn',
            title="Event Confidence Heatmap"
        )
        
        fig.update_layout(template="plotly_dark")
        return fig
    
    def create_summary_table(self, events: List[MarketEvent]) -> pd.DataFrame:
        """Create a summary table of all detected events."""
        if not events:
            return pd.DataFrame()
            
        summary_data = []
        for event in events:
            summary_data.append({
                'Timestamp': event.timestamp,
                'Type': event.event_type.value,
                'Direction': event.direction,
                'Price': f"{event.price:.2f}",
                'Confidence': f"{event.confidence:.2f}",
                'Broken_Level': f"{event.broken_level['price']:.2f}",
                'Description': event.description[:50] + "..." if len(event.description) > 50 else event.description
            })
        
        return pd.DataFrame(summary_data)

# --- Usage Example ---
def plot_example():
    """Example of how to use the enhanced plotting."""
    
    # Sample OHLC data (you would load your actual data here)
    sample_ohlc = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'open': [3300 + i*0.5 for i in range(100)],
        'high': [3305 + i*0.5 for i in range(100)],
        'low': [3295 + i*0.5 for i in range(100)],
        'close': [3302 + i*0.5 for i in range(100)]
    })
    
    # Sample events (from your detection logic)
    sample_events = []  # Your detected events would go here
    
    # Create plotter and generate chart
    plotter = EnhancedChartPlotter()
    fig = plotter.create_enhanced_chart(sample_ohlc, sample_events)
    
    # Show the chart
    fig.show()
    
    # Create confidence heatmap
    heatmap_fig = plotter.create_confidence_heatmap(sample_events)
    heatmap_fig.show()
    
    # Create summary table
    summary_df = plotter.create_summary_table(sample_events)
    print(summary_df)

if __name__ == "__main__":
    plot_example()