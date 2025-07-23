import matplotlib.pyplot as plt

def plot_levels(df, bos_choch_type, level_points, tf_name="1H"):
    # Use index for time axis if 'timestamp' column doesn't exist
    time_col = df.index if "timestamp" not in df.columns else df["timestamp"]

    plt.figure(figsize=(14, 6))
    plt.plot(time_col, df["close"], label="Price", linewidth=1.2)

    colors = {
        "BOS": "green",
        "CHOCH": "red"
    }

    for level in level_points:
        ts = level["timestamp"]
        price = float(level["price"])
        color = colors.get(bos_choch_type.upper(), "blue")
        plt.axhline(price, linestyle="--", color=color, linewidth=1)
        plt.text(df.index[-1], price, f"{bos_choch_type} @ {price:.5f}", color=color, fontsize=9, va='bottom')

    plt.title(f"{tf_name} Chart with {bos_choch_type} Levels")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
