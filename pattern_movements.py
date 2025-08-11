import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.pivot_detector import find_pivots

def apply_base_chart_style(fig: go.Figure, df: pd.DataFrame, title: str, price_col: str = 'close', height: int = 700):
    """
    Applies a consistent, professional style to a Plotly chart object.

    Args:
        fig: The Plotly Figure object to be styled.
        df: The DataFrame used for the chart (to set the price range).
        title: The title for the chart.
        price_col: The name of the price column in the DataFrame.
        height: The height of the chart in pixels.
    """
    # === 1. Update Layout (The main styling block) ===
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
        height=height,
        xaxis_rangeslider_visible=False # Hide the range slider for a cleaner look
    )

    # === 2. Update Axes Styles ===
    # Style x-axis
    fig.update_xaxes(
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True
    )

    # Style y-axis
    price_range = df[price_col].max() - df[price_col].min()
    price_padding = price_range * 0.05
    price_min = df[price_col].min() - price_padding
    price_max = df[price_col].max() + price_padding
    
    fig.update_yaxes(
        title_text="Price",
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        zeroline=False,
        range=[price_min, price_max]
    )

    return fig

# --- 1. Load and Prepare Data ---
# Use XAUUSD H1 data from your data directory with proper column handling
try:
    # Load data without header, using the expected column names
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    ohlc = pd.read_csv('data/XAUUSD_H1.csv', names=cols, header=None)
    
    # Convert datetime and set as index
    ohlc["datetime"] = pd.to_datetime(ohlc["datetime"], utc=True)
    ohlc = ohlc.set_index("datetime").astype(float)
    
    # Rename columns to capitalized format for pivot detector compatibility
    ohlc = ohlc.rename(columns={
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    print(f"Loaded XAUUSD H1 data: {len(ohlc)} records from {ohlc.index[0]} to {ohlc.index[-1]}")
except Exception as e:
    print(f"Failed to load XAUUSD data: {e}")
    exit()

# Filter for a recent date range (last 7 days of data)
df = ohlc.tail(168)  # 168 hours = 7 days

# --- 2. Detect Pivot Points ---
# Adjusted prominence for XAUUSD (gold typically has smaller price movements)
pivots = find_pivots(df, prominence=0.001)

# --- 3. Prepare for Plotting ---
# Create the list of points for the white dashed zig-zag line
zigzag_points = list(zip(pivots.index, pivots['price']))

# --- 4. Create Plotly Chart ---
# Create single plot for price chart
fig = go.Figure()

# Add candlestick chart
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='XAUUSD H1',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350'
    )
)

# Add pivot point markers
fig.add_trace(
    go.Scatter(
        x=pivots.index,
        y=pivots['price'],
        mode='markers',
        marker=dict(
            color='red',
            size=8,
            symbol='circle'
        ),
        name='Pivot Points',
        showlegend=True
    )
)

# Add zigzag line connecting pivot points
if len(zigzag_points) > 1:
    zigzag_x = [point[0] for point in zigzag_points]
    zigzag_y = [point[1] for point in zigzag_points]
    
    fig.add_trace(
        go.Scatter(
            x=zigzag_x,
            y=zigzag_y,
            mode='lines',
            line=dict(
                color='black',
                width=1.2,
                dash='dash'
            ),
            name='Zigzag Pattern',
            showlegend=True
        )
    )



# Apply the professional styling
fig = apply_base_chart_style(
    fig=fig,
    df=df,
    title='XAUUSD H1 - Price Movements with Pivot Points (Last 60 Days)',
    price_col='Close',
    height=700
)

# Show the chart
fig.show(renderer="browser")