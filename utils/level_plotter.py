import pandas as pd
import mplfinance as mpf
from typing import List, Dict

def plot_market_events(
    df: pd.DataFrame,
    events: List[Dict],
    symbol: str = "Symbol",
    tf_name: str = "Timeframe",
    save_path: str = None
):
    """
    Plots a candlestick chart with BOS/CHOCH events and non-overlapping labels for key levels.
    """
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_columns) or df.empty:
        print(f"❌ Plotting failed. DataFrame must not be empty and must contain: {required_columns}")
        return

    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit')
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', facecolor='#161A25', gridcolor='#474D59')
    
    fig, axes = mpf.plot(
        df, type='candle', style=style,
        title=f"{symbol} - {tf_name} | Market Events",
        ylabel='Price', figsize=(16, 8), returnfig=True
    )
    ax = axes[0]

    # --- New Label Management Logic ---
    label_positions = []
    y_range = df['High'].max() - df['Low'].min()
    min_label_distance = y_range * 0.04 # Minimum vertical distance between labels

    def get_label_y_pos(target_y, existing_positions):
        """Adjusts label Y-position to avoid overlap."""
        new_y = target_y
        for pos_y in existing_positions:
            if abs(new_y - pos_y) < min_label_distance:
                new_y = pos_y + min_label_distance # Stack the new label on top
        return new_y

    def draw_zone(level_name: str, level_data: Dict, zone_color: str):
        if not level_data: return
        price = level_data['price']
        
        # Adjust label position to prevent overlap
        adjusted_y = get_label_y_pos(price, label_positions)
        label_positions.append(adjusted_y)

        # Draw the shaded zone
        ax.fill_between(df.index, price * 0.999, price * 1.001, facecolor=zone_color, alpha=0.1)
        ax.axhline(price, color=zone_color, linestyle=':', linewidth=0.7, alpha=0.5)
        
        # Add a line from the level price to the spaced-out label
        ax.plot([df.index[-10], df.index[-1]], [price, adjusted_y], color=zone_color, linestyle='--', linewidth=0.6, alpha=0.7)
        ax.text(df.index[-1], adjusted_y, f' {level_name} @ {price:.2f} ', color='white', ha='left', va='center',
                bbox=dict(facecolor=zone_color, alpha=0.9, pad=1.5), fontsize=8)

    # --- Draw Events and Levels ---
    for event in events:
        event_type, direction = event['type'], event['direction']
        event_time, event_price = event['timestamp'], event['price']

        color_map = {('BOS', 'Bullish'): '#26a69a', ('BOS', 'Bearish'): '#ef5350',
                     ('CHOCH', 'Bullish'): '#2196F3', ('CHOCH', 'Bearish'): '#FFA726'}
        border_color = color_map.get((event_type, direction), 'grey')
        
        ax.axvline(event_time, color=border_color, linestyle='--', linewidth=1.2)
        ax.text(event_time, event_price, f' {event_type} ', color='white',
                fontweight='bold', bbox=dict(facecolor=border_color, alpha=0.9, pad=1))
        
        # Collect and draw all levels associated with this event
        levels_to_draw = []
        if event_type == 'BOS':
            levels_to_draw = [('TJL1', event.get('tjl1')), ('A+ Entry', event.get('tjl2_a_plus'))]
        elif event_type == 'CHOCH':
            levels_to_draw = [('SBR/RBS', event.get('sbr_level') or event.get('rbs_level')),
                              ('QML (A+)', event.get('qml_a_plus'))]
        
        # Sort levels by price to prevent line crossings
        sorted_levels = sorted([lvl for lvl in levels_to_draw if lvl[1]], key=lambda x: x[1]['price'])
        for name, data in sorted_levels:
            draw_zone(name, data, border_color)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    else:
        mpf.show()