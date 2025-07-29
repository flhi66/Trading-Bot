import matplotlib.pyplot as plt

def plot_levels(df, level_points, tf_name="1H"):
    time_col = df.index if "timestamp" not in df.columns else df["timestamp"]

    plt.figure(figsize=(14, 6))
    plt.plot(time_col, df["close"], label="Price", linewidth=1.2)

    for level in level_points:
        price = float(level["price"])
        type_ = level.get("type", "").upper()
        direction = level.get("direction", "").lower()

        # === Main BOS/CHOCH line setup ===
        if type_ == "BOS":
            color = "green" if direction == "bullish" else "red"
            label = f"BOS {direction.capitalize()} @ {price:.2f}"
        elif type_ == "CHOCH":
            color = "blue" if direction == "bullish" else "orange"
            label = f"CHOCH {direction.capitalize()} @ {price:.2f}"
        else:
            color = "gray"
            label = f"{type_} @ {price:.2f}"

        plt.axhline(price, linestyle="--", color=color, linewidth=1)
        plt.text(df.index[-1], price, label, color=color, fontsize=8, va='bottom')

        # === Extra BOS entry zones (TJL1 & TJL2) ===
        if "tjl1" in level and level["tjl1"]:
            tjl1_price = float(level["tjl1"]["price"])
            tjl1_time = level["tjl1"]["timestamp"]
            plt.axhline(tjl1_price, linestyle="--", color="purple", linewidth=0.8)
            plt.text(df.index[-1], tjl1_price, f"TJL1 @ {tjl1_price:.2f}", color="purple", fontsize=7, va='bottom')
            plt.axvline(tjl1_time, color='purple', linestyle='--', linewidth=0.5)

        if "tjl2" in level and level["tjl2"]:
            tjl2_price = float(level["tjl2"]["price"])
            tjl2_time = level["tjl2"]["timestamp"]
            plt.axhline(tjl2_price, linestyle="--", color="magenta", linewidth=0.8)
            plt.text(df.index[-1], tjl2_price, f"TJL2 @ {tjl2_price:.2f}", color="magenta", fontsize=7, va='bottom')
            plt.axvline(tjl2_time, color='magenta', linestyle='--', linewidth=0.5)

    plt.title(f"{tf_name} Chart with BOS & CHOCH Levels")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
