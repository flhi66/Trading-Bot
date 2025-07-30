from core.data_loader import load_and_resample
from core.structure_builder import get_market_analysis
from core.smart_money_concepts import MarketStructureAnalyzer
from utils.level_plotter import EnhancedChartPlotter
import pandas as pd

# === Step 1: Load data ===
symbol = "XAUUSD_H1.csv"
resampled = load_and_resample(f"data/{symbol}")
h1_data = resampled.get("1H")

if h1_data is None or h1_data.empty:
    print(f"❌ ERROR: No data loaded for the '1H' timeframe.")
else:
    # === Step 2: Get base market structure ===
    analysis = get_market_analysis(h1_data)
    structure = analysis['structure']

    # === Step 3: Use the analyzer to detect events ===
    analyzer = MarketStructureAnalyzer(confidence_threshold=0.5)
    all_events = analyzer.get_market_events(structure)
    print(f"Found {len(all_events)} total events with confidence >= {analyzer.confidence_threshold}")

    # === Step 4: Generate and show both plots ===
    if all_events:
        print("\n✅ Generating plots...")
        plotter = EnhancedChartPlotter()
        
        # --- Plot 1: Show only the event detections ---
        fig_detections = plotter.plot_event_detections(
            df=h1_data,
            events=all_events,
            symbol=symbol.split('_')[0]
        )
        print("-> Displaying Chart 1: Event Detections")
        fig_detections.show()

        # --- Plot 2: Show only the resulting trading levels ---
        fig_levels = plotter.plot_event_levels(
            df=h1_data,
            events=all_events,
            symbol=symbol.split('_')[0]
        )
        print("-> Displaying Chart 2: Key Trading Levels")
        fig_levels.show()

    else:
        print("\n✅ Analysis complete. No high-confidence market events to plot.")