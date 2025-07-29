from core.data_loader import load_and_resample
from core.structure_builder import get_structure
from core.level_marker import mark_bos_choch_levels
from utils.level_plotter import plot_levels

# === Step 1: Load and resample data ===
symbol = "XAUUSD_H1.csv"
resampled = load_and_resample(f"data/{symbol}")
h1_data = resampled["1H"]

# === Step 2: Get structure and trend ===
structure, trend_info = get_structure(h1_data)
trend = trend_info["trend"] if isinstance(trend_info, dict) and "trend" in trend_info else trend_info

# === Step 3: Detect BOS & CHOCH levels ===
bos_choch_result = mark_bos_choch_levels(structure, trend)
bos_list = bos_choch_result.get("bos", [])
choch_list = bos_choch_result.get("choch", [])

def safe_direction(pt):
    return pt.get("direction", "unknown")

# === Step 4: Filter and count by direction ===
bos_bullish = [pt for pt in bos_list if safe_direction(pt) == 'bullish']
bos_bearish = [pt for pt in bos_list if safe_direction(pt) == 'bearish']
choch_bullish = [pt for pt in choch_list if safe_direction(pt) == 'bullish']
choch_bearish = [pt for pt in choch_list if safe_direction(pt) == 'bearish']

# === Step 5: Print stats ===
print(f"Trend: {trend}")
print(f"✅ BOS Bullish: {len(bos_bullish)}")
print(f"✅ BOS Bearish: {len(bos_bearish)}")
print(f"✅ CHOCH Bullish: {len(choch_bullish)}")
print(f"✅ CHOCH Bearish: {len(choch_bearish)}")

# === Step 6: Combine and tag for plotting ===
level_points = []

for pt in bos_list:
    pt["type"] = "BOS"
    level_points.append(pt)

for pt in choch_list:
    pt["type"] = "CHOCH"
    level_points.append(pt)

# === Step 7: Plot all BOS and CHOCH levels ===
if level_points:
    level_points = level_points[-10:]  # show only last 20 levels
    plot_levels(
        df=h1_data,
        level_points=level_points,
        tf_name="1H"
    )
else:
    print("❌ No BOS or CHOCH levels detected to plot.")
