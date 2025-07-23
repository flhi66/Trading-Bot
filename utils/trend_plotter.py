import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_trend(
    df: pd.DataFrame,
    swing_highs: list[tuple[pd.Timestamp, float]],
    swing_lows: list[tuple[pd.Timestamp, float]],
    tf_name: str = "Timeframe",
    trend_direction: str = "",
    save_path: str | None = None,
    bos_choch_type: str | None = None,
    level_points: dict | None = None
):
    """
    Plots the close price with swing highs, lows, and optional BOS/CHOCH levels.
    
    Parameters:
        df (pd.DataFrame): OHLCV data with datetime index.
        swing_highs (list): List of (timestamp, price) for swing highs.
        swing_lows (list): List of (timestamp, price) for swing lows.
        tf_name (str): Label for the timeframe (e.g., '1H', '4H').
        trend_direction (str): 'uptrend', 'downtrend', or 'sideways'.
        save_path (str | None): Save the plot if path is given.
        bos_choch_type (str | None): 'BOS' or 'CHOCH' if level to highlight.
        level_points (dict | None): Dictionary like {'A': (ts, price), ...}
    """

    df.index = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.2)

    # === Plot swing highs
    if swing_highs:
        times, prices = zip(*swing_highs)
        ax.scatter(times, prices, color='green', marker='^', label='Swing High')
        for i in range(1, len(swing_highs)):
            ax.annotate("", xy=swing_highs[i], xytext=swing_highs[i-1],
                        arrowprops=dict(arrowstyle="->", color='green', lw=1.5))

    # === Plot swing lows
    if swing_lows:
        times, prices = zip(*swing_lows)
        ax.scatter(times, prices, color='red', marker='v', label='Swing Low')
        for i in range(1, len(swing_lows)):
            ax.annotate("", xy=swing_lows[i], xytext=swing_lows[i-1],
                        arrowprops=dict(arrowstyle="->", color='red', lw=1.5))

    # === Plot BOS/CHOCH levels
    if bos_choch_type and level_points:
        level_color = 'orange' if bos_choch_type == 'CHOCH' else 'blue'
        for label, (ts, price) in level_points.items():
            ax.scatter(ts, price, color=level_color, s=100, label=f'{bos_choch_type} - {label}', zorder=5, edgecolors='black')
            ax.text(ts, price, f"{label}", color=level_color, fontsize=10, fontweight='bold', ha='center', va='bottom')

    # === Trend label placement
    latest_time = df.index[-1]
    latest_price = df['close'].iloc[-1]

    ax.text(latest_time, latest_price * 0.98,
            f"Trend: {trend_direction.upper()}\n({latest_time.strftime('%Y-%m-%d %H:%M')})",
            fontsize=12, color='blue', weight='bold',
            ha='right', va='top')

    # === Title and formatting
    trend_colors = {"uptrend": "green", "downtrend": "red", "sideways": "gray"}
    title_color = trend_colors.get(trend_direction.lower(), "blue")

    ax.set_title(f"{tf_name} Trend: {trend_direction.upper()}", fontsize=14, color=title_color)
    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()

    # Format x-axis
    if hasattr(df.index, 'freqstr') and df.index.freqstr == 'H' or tf_name == '1H':
        formatter = mdates.DateFormatter('%b %d\n%H:%M')
    elif tf_name in ['4H', 'D1']:
        formatter = mdates.DateFormatter('%b %d')
    else:
        formatter = mdates.DateFormatter('%m-%d %H:%M')

    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.tick_params(axis='x', which='major', labelsize=9)

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
