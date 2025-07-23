from core.data_loader import load_and_resample
from core.structure_builder import get_structure
from utils.trend_plotter import plot_trend

# === Step 1: Load and resample data ===
symbol = "XAUUSD_H1.csv"
resampled = load_and_resample(f"data/{symbol}")
h1_data = resampled["1H"]

# === Step 2: Detect structure and trend ===
structure, trend_info = get_structure(h1_data)

# === Step 3: Normalize trend (handle dict or string)
trend_direction = trend_info["trend"] if isinstance(trend_info, dict) and "trend" in trend_info else trend_info

# === Step 4: Extract swing points ===
swing_highs = [(pt['timestamp'], pt['price']) for pt in structure if pt['type'] in ['HH', 'LH']]
swing_lows  = [(pt['timestamp'], pt['price']) for pt in structure if pt['type'] in ['LL', 'HL']]

# === Step 5: Plot structure and trend ===
plot_trend(
    df=h1_data,
    swing_highs=swing_highs,
    swing_lows=swing_lows,
    tf_name="1H",
    trend_direction=trend_direction,
    bos_choch_type=None,
    level_points=None
)
