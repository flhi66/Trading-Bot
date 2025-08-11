import pandas as pd
import mplfinance as mpf
from core.pivot_detector import find_pivots

# --- 1. Load and Prepare Data ---
# Use XAUUSD H1 data from your data directory with proper column handling
try:
    # Load data without header, using the expected column names
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    ohlc = pd.read_csv('data/XAUUSD_H1.csv', names=cols, header=None)
    
    # Convert datetime and set as index
    ohlc["datetime"] = pd.to_datetime(ohlc["datetime"], utc=True)
    ohlc = ohlc.set_index("datetime").astype(float)
    
    # Rename columns to capitalized format for mplfinance
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

# Create the data for the cyan pivot markers - ensure alignment with main dataframe
pivot_markers = pd.Series(index=df.index, dtype=float)  # Create series with same index as df
pivot_markers.loc[pivots.index] = pivots['price']  # Fill only the pivot points
pivot_markers = pivot_markers.fillna(float('nan'))  # Fill missing values with NaN

ap = [
    mpf.make_addplot(pivot_markers, type='scatter', color='cyan', marker='o', markersize=25)
]

# --- 4. Define the Chart Style ---
# This style closely matches the one in your image
mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit')
style = mpf.make_mpf_style(
    base_mpf_style='nightclouds', 
    marketcolors=mc,
    gridstyle='--',
    gridcolor='#404040'  # Fixed: using hex color instead of rgba
)

# --- 5. Create and Show the Chart ---
fig, axes = mpf.plot(
    df,
    type='candle',
    style=style,
    title='XAUUSD H1 - Price Movements with Pivot Points',
    ylabel='Price (USD)',
    figsize=(12, 5),
    alines=dict(alines=zigzag_points, colors='white', linestyle='--', linewidths=1.2),
    addplot=ap,
    returnfig=True
)

# Customize axis labels to match the minimalist look
axes[0].set_xlabel('')
axes[0].legend(['XAUUSD_H1']) # Updated legend for XAUUSD

mpf.show()