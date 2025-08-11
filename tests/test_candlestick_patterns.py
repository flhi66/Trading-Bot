from core.data_loader import load_and_resample
from core.candlestick_patterns import CandlestickPatternDetector, PatternType
from utils.candlestick_plotter import CandlestickPatternPlotter
import pandas as pd

def remove_duplicate_patterns(patterns, time_threshold_minutes=60, price_threshold_pct=0.1):
    """Remove duplicate patterns that are too close in time and price"""
    if not patterns:
        return patterns
    
    # Sort patterns by timestamp
    sorted_patterns = sorted(patterns, key=lambda x: x.timestamp)
    filtered_patterns = []
    
    for pattern in sorted_patterns:
        is_duplicate = False
        
        for existing in filtered_patterns:
            # Check if patterns are of same type
            if pattern.pattern_type == existing.pattern_type:
                # Check time difference
                time_diff = abs((pattern.timestamp - existing.timestamp).total_seconds() / 60)
                
                # Check price difference
                price_diff = abs(pattern.price - existing.price) / existing.price * 100
                
                # If too close in time and price, consider duplicate
                if time_diff <= time_threshold_minutes and price_diff <= price_threshold_pct:
                    # Keep the one with higher confidence
                    if pattern.confidence > existing.confidence:
                        # Remove the existing lower confidence pattern
                        filtered_patterns.remove(existing)
                        break
                    else:
                        # Skip this pattern as it's a lower confidence duplicate
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            filtered_patterns.append(pattern)
    
    return filtered_patterns

def generate_entry_levels(df, patterns):
    """Generate entry levels based on detected patterns"""
    entry_levels = []
    
    for pattern in patterns:
        # Get the pattern candle(s)
        pattern_idx = df.index.get_loc(pattern.timestamp)
        pattern_candle = df.iloc[pattern_idx]
        
        # Generate entry level based on pattern type
        if pattern.direction == 'Bullish':
            # For bullish patterns, entry slightly above the high
            entry_price = pattern_candle['High'] + (pattern_candle['High'] - pattern_candle['Low']) * 0.1
            stop_loss = pattern_candle['Low'] - (pattern_candle['High'] - pattern_candle['Low']) * 0.2
            take_profit = entry_price + (entry_price - stop_loss) * 2  # 1:2 RR
        elif pattern.direction == 'Bearish':
            # For bearish patterns, entry slightly below the low
            entry_price = pattern_candle['Low'] - (pattern_candle['High'] - pattern_candle['Low']) * 0.1
            stop_loss = pattern_candle['High'] + (pattern_candle['High'] - pattern_candle['Low']) * 0.2
            take_profit = entry_price - (stop_loss - entry_price) * 2  # 1:2 RR
        else:  # Neutral patterns
            # For neutral patterns like Doji, wait for breakout
            entry_price = pattern_candle['Close']
            stop_loss = pattern_candle['Low'] if pattern_candle['Close'] > pattern_candle['Open'] else pattern_candle['High']
            take_profit = entry_price + (entry_price - stop_loss) * 1.5  # 1:1.5 RR
        
        entry_levels.append({
            'timestamp': pattern.timestamp,
            'pattern_type': pattern.pattern_type,
            'direction': pattern.direction,
            'confidence': pattern.confidence,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        })
    
    return entry_levels

def main():
    # === Step 1: Load data ===
    symbol = "XAUUSD_H1.csv"
    resampled = load_and_resample(f"data/{symbol}")
    h1_data = resampled.get("1H")

    if h1_data is None or h1_data.empty:
        print(f"❌ ERROR: No data loaded for the '1H' timeframe.")
        return

    print(f"✅ Loaded {len(h1_data)} candles for {symbol}")

    # === Step 2: Detect candlestick patterns ===
    detector = CandlestickPatternDetector()
    all_patterns = detector.detect_patterns(h1_data, min_confidence=0.6)
    # Filter for only high confidence patterns (95%+)
    high_confidence_patterns = [p for p in all_patterns if p.confidence >= 0.95]
    # Remove duplicates
    patterns = remove_duplicate_patterns(high_confidence_patterns)
    
    print(f"\n🔍 Detected {len(patterns)} unique high-confidence (95%+) candlestick patterns:")
    print(f"    (Filtered from {len(high_confidence_patterns)} total high-confidence patterns)")
    
    # Group patterns by type
    pattern_counts = {}
    for pattern in patterns:
        if pattern.pattern_type not in pattern_counts:
            pattern_counts[pattern.pattern_type] = []
        pattern_counts[pattern.pattern_type].append(pattern)
    
    # Display pattern summary
    for pattern_type, pattern_list in pattern_counts.items():
        print(f"  - {pattern_type.value}: {len(pattern_list)} patterns")
    
    # Show details of high-confidence patterns
    high_confidence_patterns = [p for p in patterns if p.confidence >= 0.8]
    if high_confidence_patterns:
        print(f"\n📊 High-confidence patterns (>80%):")
        for pattern in high_confidence_patterns[:10]:  # Show top 10
            print(f"  - {pattern.pattern_type.value} @ ${pattern.price:.2f} "
                  f"({pattern.confidence:.0%}) - {pattern.direction}")

    # === Step 3: Create charts ===
    if patterns:
        print("\n✅ Generating candlestick pattern charts...")
        plotter = CandlestickPatternPlotter()
        
        # Chart 1: Overview of all patterns
        fig_overview = plotter.plot_patterns_overview(
            df=h1_data,
            patterns=patterns,
            symbol=symbol.split('_')[0],
            min_confidence=0.6
        )
        print("-> Displaying Chart 1: Patterns Overview")
        fig_overview.show()

        # Chart 2: Bullish patterns only
        fig_bullish = plotter.plot_bullish_patterns(
            df=h1_data,
            patterns=patterns,
            symbol=symbol.split('_')[0]
        )
        print("-> Displaying Chart 2: Bullish Patterns")
        fig_bullish.show()

        # Chart 3: Bearish patterns only
        fig_bearish = plotter.plot_bearish_patterns(
            df=h1_data,
            patterns=patterns,
            symbol=symbol.split('_')[0]
        )
        print("-> Displaying Chart 3: Bearish Patterns")
        fig_bearish.show()

        # Chart 4: Pattern analysis with frequency
        fig_analysis = plotter.plot_pattern_analysis(
            df=h1_data,
            patterns=patterns,
            symbol=symbol.split('_')[0]
        )
        print("-> Displaying Chart 4: Pattern Analysis")
        fig_analysis.show()

        # Chart 5: Entry Levels Based on Pattern Detection (Top 50 patterns)
        if patterns:
            try:
                print("-> Generating entry levels...")
                # Limit to top 50 patterns by confidence for better visualization
                top_patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)[:50]
                entry_levels = generate_entry_levels(h1_data, top_patterns)
                print(f"-> Generated {len(entry_levels)} entry levels (top 50 by confidence)")
                
                print("-> Creating entry levels chart...")
                fig_entries = plotter.plot_entry_levels(
                    df=h1_data,
                    patterns=top_patterns,
                    entry_levels=entry_levels,
                    symbol=symbol.split('_')[0]
                )
                print(f"-> Displaying Chart 5: Entry Levels ({len(entry_levels)} levels)")
                fig_entries.show()
                print("-> Chart 5 displayed successfully")
            except Exception as e:
                print(f"❌ Error generating Chart 5: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("-> No patterns available for Chart 5")

    else:
        print("\n✅ Analysis complete. No candlestick patterns detected with the current criteria.")

if __name__ == "__main__":
    main()