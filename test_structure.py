from core.data_loader import load_and_resample
from core.structure_builder import get_structure
from core.level_marker import detect_bos_choch

# === Step 1: Load and resample data ===
symbol = "GBPUSD_H1.csv"  # Ensure this file exists in /data
resampled = load_and_resample(f"data/{symbol}")
h1_data = resampled["1H"]

# === Step 2: Detect structure and trend ===
structure, trend = get_structure(h1_data)

# === Step 3: Show recent structure ===
print("\n--- Last 6 Structure Points ---")
for pt in structure[-6:]:
    print(pt)

print("\n📌 Confirmed Structure Trend:", trend)

# === Step 4: Detect BOS or CHOCH ===
bos_choch_type, levels = detect_bos_choch(structure, trend, window=30)

# === Step 5: Display BOS/CHOCH detection ===
if bos_choch_type:
    print(f"\n✅ {bos_choch_type} detected!")
    print("🔍 Level Points:")
    for i, pt in enumerate(levels):
        label = chr(65 + i)  # A, B, C...
        print(f"  {label}: {pt['timestamp']} — {pt['price']}")
else:
    print("\n❌ No BOS or CHOCH detected.")

