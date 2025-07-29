import pandas as pd
import mplfinance as mpf
from typing import List, Dict

def plot_market_structure(
    df: pd.DataFrame,
    structure: List[Dict],
    trend_direction: str,
    symbol: str = "Symbol",
    tf_name: str = "Timeframe",
    save_path: str = None
):
    """
    Plots a professional candlestick chart showing the market structure (HH, HL, LH, LL).
    """
    if df.empty or not structure:
        print("Data or structure is empty, cannot plot.")
        return
        
    # --- 1. Define a professional-looking style ---
    mc = mpf.make_marketcolors(
        up='tab:green', down='tab:red',
        edge={'up':'#00B746', 'down':'#F03E3E'},
        wick={'up':'#00B746', 'down':'#F03E3E'},
        volume='inherit'
    )
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', facecolor='#161A25', gridcolor='#474D59')

    # --- 2. Prepare plot data and annotations ---
    sorted_structure = sorted(structure, key=lambda x: x['timestamp'])
    
    up_points = {p['timestamp']: p['price'] for p in sorted_structure if p['type'] in ['HH', 'HL']}
    down_points = {p['timestamp']: p['price'] for p in sorted_structure if p['type'] in ['LH', 'LL']}
    
    up_series = pd.Series(up_points).reindex(df.index)
    down_series = pd.Series(down_points).reindex(df.index)

    add_plots = [
        mpf.make_addplot(up_series, type='scatter', color='#26a69a', marker='o', markersize=50),
        mpf.make_addplot(down_series, type='scatter', color='#ef5350', marker='o', markersize=50),
    ]

    structure_line_points = [(p['timestamp'], p['price']) for p in sorted_structure]
    alines = dict(alines=[structure_line_points], colors=['#FFD700'], linewidths=1.2)
    
    # --- 3. Create the plot title ---
    trend_colors = {"uptrend": "#26a69a", "downtrend": "#ef5350", "sideways": "#787b86"}
    title_color = trend_colors.get(trend_direction.lower(), "#5682A3")
    title = f"{symbol} - {tf_name} | Trend: {trend_direction.upper()}"
    
    # --- 4. Generate the plot and get access to the axes ---
    fig, axes = mpf.plot(
        df, type='candle', style=style, title=title, ylabel='Price',
        addplot=add_plots, alines=alines, figsize=(15, 7), returnfig=True
    )

    # --- 5. Add HH/HL/LH/LL text labels with background boxes for visibility ---
    y_range = df['high'].max() - df['low'].min()
    offset = y_range * 0.03

    for point in sorted_structure:
        point_type = point['type']
        vertical_align = 'bottom' if point_type in ['HH', 'LH'] else 'top'
        y_pos = point['price'] + offset if vertical_align == 'bottom' else point['price'] - offset
        
        # Define label color based on type
        label_color = '#26a69a' if point_type in ['HH', 'HL'] else '#ef5350'
        
        # Add a background box to the text to make it stand out
        bbox_props = dict(boxstyle='round,pad=0.2', facecolor=label_color, alpha=0.7)
        
        axes[0].text(
            point['timestamp'], y_pos, point_type,
            ha='center', va='center',  # Center text inside the box
            fontsize=9, fontweight='bold',
            color='white',
            bbox=bbox_props # Apply the background box
        )

    # Add a clean text box for the trend in the top-left corner
    axes[0].text(0.02, 0.95, f"TREND: {trend_direction.upper()}",
               transform=axes[0].transAxes,
               fontweight='bold', fontsize=12,
               color='white', backgroundcolor=title_color,
               va='top', ha='left')

    # --- 6. Save or show the plot ---
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        mpf.show()