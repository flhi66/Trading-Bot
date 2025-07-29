from core.data_loader import load_and_resample
from core.structure_builder import get_market_analysis
from utils.trend_plotter import plot_market_structure

# === Step 1: Load and resample data ===
symbol = "XAUUSD_H1.csv"
resampled = load_and_resample(f"data/{symbol}")
h1_data = resampled.get("1H")

if h1_data is None or h1_data.empty:
    print(f"Could not load data for {symbol}. Exiting.")
else:
    # === Step 2: Get the complete market analysis on the full dataset ===
    analysis = get_market_analysis(h1_data)
    
    # === Step 3: Select recent data for a clean plot ===
    recent_data = h1_data.iloc[-150:]
    
    # === FIX: Filter the structure points to match the recent data's date range ===
    start_date = recent_data.index[0]
    end_date = recent_data.index[-1]
    
    recent_structure = [
        p for p in analysis['structure'] 
        if start_date <= p['timestamp'] <= end_date
    ]
    # === End of fix ===
    
    # === Step 4: Plot the market structure using the filtered data ===
    plot_market_structure(
        df=recent_data,
        structure=recent_structure, # Pass the filtered structure
        trend_direction=analysis['trend'],
        symbol=symbol.split('_')[0],
        tf_name="1H",
        save_path=None
    )