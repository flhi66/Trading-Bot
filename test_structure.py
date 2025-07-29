import pandas as pd
from core.data_loader import load_and_resample
from core.structure_builder import get_market_analysis

# === Step 1: Load and resample data ===
symbol = "XAUUSD_H1.csv"  # Ensure this file exists in the /data folder
resampled = load_and_resample(f"data/{symbol}")
h1_data = resampled.get("1H")

if h1_data is None or h1_data.empty:
    print(f"Could not load data for {symbol}. Exiting.")
else:
    # === Step 2: Get the complete market analysis ===
    # This single function provides the trend and detailed structure points (HH, HL, etc.)
    market_analysis = get_market_analysis(h1_data)

    # === Step 3: Display the results ===
    print("\n✅ Market Analysis Results")
    print("="*30)
    
    # Print the final determined trend
    trend = market_analysis.get('trend', 'unknown')
    print(f"Detected Trend: {trend.upper()}")
    
    # Print the last 10 identified structure points
    structure_points = market_analysis.get('structure', [])
    if structure_points:
        print("\nRecent Market Structure (HH, HL, LH, LL):")
        for point in structure_points[-10:]: # Show last 10 points
            timestamp = point.get('timestamp').strftime('%Y-%m-%d %H:%M')
            pt_type = point.get('type')
            price = point.get('price')
            print(f"  - {timestamp} | {pt_type:<3} | Price: {price:.5f}")
    else:
        print("No market structure points were detected.")
        
    print("="*30)