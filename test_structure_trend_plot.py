from core.data_loader import load_and_resample
from core.structure_builder import get_market_analysis
from core.trend_detector import detect_swing_points, detect_trend
from utils.structure_trend_plotter import StructureTrendPlotter
import pandas as pd

# === Step 1: Load data ===
symbol = "XAUUSD_H1.csv"
resampled = load_and_resample(f"data/{symbol}")
h1_data = resampled.get("1H")

if h1_data is None or h1_data.empty:
    print(f"❌ ERROR: No data loaded for the '1H' timeframe.")
else:
    print(f"✅ Loaded {len(h1_data)} candles for {symbol}")
    
    # === Step 2: Get base analysis ===
    analysis = get_market_analysis(h1_data)
    structure = analysis['structure']
    
    # Detect swing points for trend analysis
    df_lower = h1_data.copy()
    df_lower.columns = [col.lower() for col in df_lower.columns]
    swing_highs, swing_lows = detect_swing_points(df_lower)
    trend = detect_trend(swing_highs, swing_lows)
    
    print(f"Found {len(structure)} structure points")
    print(f"Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
    print(f"Detected trend: {trend}")

    # === Step 3: Generate and show two plots ===
    if structure or (swing_highs and swing_lows):
        print("\n✅ Generating plots...")
        plotter = StructureTrendPlotter()
        
        # Debug: Check what points are being detected
        structure_counts = {}
        for point in structure:
            point_type = point['type']
            structure_counts[point_type] = structure_counts.get(point_type, 0) + 1
        
        print(f"🔍 Debug: Chart 1 will show {len(swing_highs) + len(swing_lows)} swing points")
        print(f"🔍 Debug: Chart 2 will show {len(structure)} structure points:")
        
        # Group by type
        for struct_type, count in structure_counts.items():
            print(f"  - {struct_type}: {count} points")
        
        # --- Plot 1: Show trend detection with swing points ---
        fig_trend = plotter.plot_trend_detection(
            df=h1_data,
            symbol=symbol.split('_')[0],
            timeframe="1H"
        )
        print("-> Displaying Chart 1: Trend Detection")
        fig_trend.show()

        # --- Plot 2: Show market structure (HH/HL/LH/LL) ---
        fig_structure = plotter.plot_structure_analysis(
            df=h1_data,
            symbol=symbol.split('_')[0],
            timeframe="1H"
        )
        print("-> Displaying Chart 2: Market Structure Analysis")
        fig_structure.show()

    else:
        print("\n✅ Analysis complete. No structure or swing points detected to plot.")