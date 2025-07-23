import pandas as pd
from core.data_loader import load_and_resample
from core.trend_detector import get_trend_from_data, detect_swing_points, detect_trend
from utils.trend_plotter import plot_trend

def test_trend_visualization(symbol: str):
    try:
        resampled_data = load_and_resample(symbol)
    except FileNotFoundError as e:
        print(f"❌ Data not found for symbol '{symbol}': {e}")
        return

    # Detect trend with override logic
    final_trend = get_trend_from_data(resampled_data)

    for tf in ["4H", "1H", "15M"]:
        df = resampled_data.get(tf)
        if df is None or df.empty:
            print(f"⚠️ No data for timeframe {tf}")
            continue

        df.index = pd.to_datetime(df.index)
        swing_highs, swing_lows = detect_swing_points(df)
        tf_trend = detect_trend(swing_highs, swing_lows)

        print(f"\n📉 {tf} Trend: {tf_trend.upper()} — Final Decision: {final_trend.upper()}")

        plot_trend(
            df=df,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            tf_name=tf,
            trend_direction=tf_trend  # more accurate than applying final trend to all
        )

if __name__ == "__main__":
    # 🔧 Change this to any symbol file you've added in /data
    test_trend_visualization("XAUUSD")
