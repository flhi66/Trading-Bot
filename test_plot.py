from core.data_loader import load_and_resample
from core.structure_builder import get_structure
from core.level_marker import detect_bos_choch
from utils.trend_plotter import plot_trend

# === Step 1: Load resampled data ===
symbol = "GBPUSD_H1.csv"
resampled = load_and_resample(f"data/{symbol}")
h1_data = resampled["1H"]

# === Step 2: Get market structure and trend ===
structure, trend = get_structure(h1_data)

# === Step 3: Detect BOS/CHOCH (updated to use tuple output)
level_type, level_points = detect_bos_choch(structure, trend, window=30)

# === Step 4: Extract swing highs and lows ===
swing_highs = [(pt['timestamp'], pt['price']) for pt in structure if pt['type'] in ['HH', 'LH']]
swing_lows  = [(pt['timestamp'], pt['price']) for pt in structure if pt['type'] in ['LL', 'HL']]

# === Step 5: Plot trend + BOS/CHOCH levels if found ===
print(level_points)
level_points_dict = {pt['type']: (pt['timestamp'], pt['price']) for pt in level_points}
plot_trend(
    df=h1_data,
    swing_highs=swing_highs,
    swing_lows=swing_lows,
    tf_name="1H",
    trend_direction=trend,
    bos_choch_type=level_type,
    level_points=level_points_dict
)
