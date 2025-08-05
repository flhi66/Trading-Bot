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

    # === Step 4: Generate and show three plots ===
    if all_events:
        print("\n✅ Generating plots...")
        plotter = EnhancedChartPlotter()
        
        # Debug: Check what levels are being detected
        qml_levels = plotter._detect_qml_levels(h1_data, all_events)
        aplus_levels = plotter._extract_aplus_levels(all_events)
        
        print(f"🔍 Debug: Detected {len(qml_levels)} QML levels")
        print(f"🔍 Debug: Detected {len(aplus_levels)} A+ levels")
        
        if aplus_levels:
            print("📊 A+ Levels found:")
            for aplus in aplus_levels:
                print(f"  - {aplus['name']} @ ${aplus['price']:.2f} ({aplus['entry_type']})")
        
        # --- Plot 1: Show only the event detections ---
        fig_detections = plotter.plot_event_detections(
            df=h1_data,
            events=all_events,
            symbol=symbol.split('_')[0]
        )
        print("-> Displaying Chart 1: Event Detections")
        fig_detections.show()

        # --- Plot 2: Show all levels (TJL, QML, A+) with horizontal lines ---
        fig_all_levels = plotter.plot_all_levels(
            df=h1_data,
            events=all_events,
            symbol=symbol.split('_')[0]
        )
        print("-> Displaying Chart 2: All Trading Levels")
        fig_all_levels.show()

        # --- Plot 3: Show only entries (triangles) without horizontal lines ---
        fig_entries_only = plotter.plot_entries_only(
            df=h1_data,
            events=all_events,
            symbol=symbol.split('_')[0]
        )
        print("-> Displaying Chart 3: Entry Points Only")
        fig_entries_only.show()

    else:
        print("\n✅ Analysis complete. No high-confidence market events to plot.")