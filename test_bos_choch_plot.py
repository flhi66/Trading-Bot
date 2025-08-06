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