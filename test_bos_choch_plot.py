from core.data_loader import load_and_resample
from core.structure_builder import get_market_analysis
from core.level_marker import get_market_events
from utils.level_plotter import plot_market_events

# === Step 1: Load data ===
symbol = "XAUUSD_H1.csv"
resampled = load_and_resample(f"data/{symbol}")
h1_data = resampled.get("1H")

if h1_data is None or h1_data.empty:
    print(f"❌ ERROR: No data loaded for the '1H' timeframe. Check your data file and loader.")
else:
    # === Step 2: Get market structure and events ===
    analysis = get_market_analysis(h1_data)
    structure = analysis['structure']
    all_events = get_market_events(structure)
    
    # === ADD THIS FOR DEBUGGING: Print the detected structure ===
    print("\n--- Detected Market Structure (Last 15 Points) ---")
    for point in structure[-15:]:
        print(point)
    print("--------------------------------------------------\n")
    # === END OF DEBUGGING ADDITION ===

    # === Step 3: Prepare data for plotting ===
    recent_data = h1_data.iloc[-250:]
    
    if not recent_data.empty:
        start_date = recent_data.index[0]
        recent_events = [e for e in all_events if e['timestamp'] >= start_date]

        # === Step 4: Plot the results ===
        if recent_events:
            print(f"\n✅ Analysis complete. Plotting {len(recent_events)} most recent market events...")
            plot_market_events(
                df=recent_data,
                events=recent_events,
                symbol=symbol.split('_')[0],
                tf_name="1H"
            )
        else:
            # This is the message you are seeing
            print("\n✅ Analysis complete. No recent market events to plot.")
    else:
        print("❌ ERROR: No recent data available to plot.")