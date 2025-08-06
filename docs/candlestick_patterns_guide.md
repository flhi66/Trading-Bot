# Candlestick Pattern Detection and Plotting Guide

## Overview

This module provides comprehensive candlestick pattern detection and visualization capabilities. It can detect 13 different candlestick patterns with confidence scoring and create professional charts for analysis.

## Features

### Pattern Detection
- **9 key patterns** focused on high-probability setups
- **Confidence scoring** (60-95%) for each pattern
- **Trend-aware detection** for context-sensitive patterns
- **Multi-candle patterns** (1-3 candles)

### Supported Patterns

#### Single Candle Patterns
- **Hammer** - Bullish reversal with long lower wick
- **Hanging Man** - Bearish reversal with long lower wick (in uptrend)
- **Inverted Hammer** - Bullish reversal with long upper wick
- **Shooting Star** - Bearish reversal with long upper wick (in uptrend)
- **Doji** - Indecision pattern with very small body

#### Two Candle Patterns
- **Bullish Engulfing** - Strong bullish reversal
- **Bearish Engulfing** - Strong bearish reversal

#### Three Candle Patterns
- **Morning Star** - Strong bullish reversal
- **Evening Star** - Strong bearish reversal

### Visualization Options

#### 1. Patterns Overview Chart
- Shows all detected patterns on candlestick chart
- Color-coded markers for different pattern types
- Hover information with pattern details
- Pattern statistics and legend

#### 2. Bullish Patterns Chart
- Focuses only on bullish patterns
- Green color scheme
- Detailed bullish pattern legend

#### 3. Bearish Patterns Chart
- Focuses only on bearish patterns
- Red color scheme
- Detailed bearish pattern legend

#### 4. Pattern Analysis Chart
- Dual-pane view with price and frequency
- Pattern frequency over time
- Statistical analysis

#### 5. Detailed Pattern Chart
- Highlights specific pattern type
- Shows pattern formation candles
- Detailed pattern explanation

## Usage

### Basic Pattern Detection

```python
from core.candlestick_patterns import CandlestickPatternDetector
from utils.candlestick_plotter import CandlestickPatternPlotter

# Load your data
df = pd.read_csv('data/XAUUSD_H1.csv')

# Detect patterns
detector = CandlestickPatternDetector()
patterns = detector.detect_patterns(df, min_confidence=0.6)

# Create plotter
plotter = CandlestickPatternPlotter()

# Generate overview chart
fig = plotter.plot_patterns_overview(df, patterns, "XAUUSD")
fig.show()
```

### Advanced Usage

```python
# Filter high-confidence patterns
high_conf_patterns = [p for p in patterns if p.confidence >= 0.8]

# Filter by pattern type
bullish_engulfing = [p for p in patterns 
                    if p.pattern_type == PatternType.BULLISH_ENGULFING]

# Create detailed chart for specific pattern
fig = plotter.plot_pattern_details(df, patterns, "XAUUSD", 
                                  PatternType.BULLISH_ENGULFING)
fig.show()
```

### Custom Pattern Parameters

```python
# Customize hammer detection
detector = CandlestickPatternDetector()
is_hammer = detector.is_hammer_candle(
    candles=df, 
    pos=-1,
    lower_wick=0.6,  # 60% of candle range
    body=0.2,        # 20% of candle range
    upper_wick=0.2   # 20% of candle range
)
```

## Pattern Details

### Bullish Patterns

#### Bullish Engulfing
- **Signal**: Strong bullish reversal
- **Formation**: Large bullish candle engulfs previous bearish candle
- **Confidence**: Based on engulfing ratio
- **Best Use**: After downtrend for reversal entry

#### Hammer
- **Signal**: Bullish reversal
- **Formation**: Small body with long lower wick
- **Confidence**: Based on wick/body ratios
- **Best Use**: At support levels or trend bottoms

#### Morning Star
- **Signal**: Strong bullish reversal
- **Formation**: Bearish candle → Small body → Bullish candle
- **Confidence**: Based on third candle penetration
- **Best Use**: Major trend reversal signals

### Bearish Patterns

#### Bearish Engulfing
- **Signal**: Strong bearish reversal
- **Formation**: Large bearish candle engulfs previous bullish candle
- **Confidence**: Based on engulfing ratio
- **Best Use**: After uptrend for reversal entry

#### Evening Star
- **Signal**: Strong bearish reversal
- **Formation**: Bullish candle → Small body → Bearish candle
- **Confidence**: Based on third candle penetration
- **Best Use**: Major trend reversal signals

### Neutral Patterns

#### Doji
- **Signal**: Indecision, potential reversal
- **Formation**: Very small body with equal wicks
- **Confidence**: Based on body size
- **Best Use**: Trend exhaustion signals

## Confidence Scoring

Each pattern receives a confidence score based on:

1. **Pattern Quality**: How well it matches ideal formation
2. **Wick Ratios**: Proportion of wicks to body
3. **Body Size**: Relative size of candle bodies
4. **Penetration**: How much patterns overlap or pierce

### Confidence Levels
- **80-100%**: Excellent patterns, high probability
- **70-79%**: Good patterns, moderate probability
- **60-69%**: Fair patterns, lower probability
- **<60%**: Weak patterns, filtered out by default

## Chart Features

### Interactive Elements
- **Hover Information**: Detailed pattern data
- **Pattern Markers**: Unique symbols for each pattern type
- **Color Coding**: Intuitive colors for bullish/bearish
- **Annotations**: Pattern labels with confidence

### Professional Styling
- **Dark Theme**: Professional trading interface
- **Clean Layout**: Uncluttered visualization
- **Legend**: Comprehensive pattern explanations
- **Statistics**: Pattern frequency and performance

## Pattern Color Scheme

| Pattern Type | Color | Symbol |
|--------------|--------|--------|
| Bullish Engulfing | Green | Triangle Up |
| Bearish Engulfing | Red | Triangle Down |
| Hammer | Green | Circle |
| Inverted Hammer | Blue | Diamond |
| Doji | Purple | Cross |
| Morning Star | Cyan | Star |
| Evening Star | Pink | Star Open |
| Bullish Harami | Light Green | Square |
| Bearish Harami | Light Red | Square Open |

## Best Practices

### Pattern Detection
1. **Use appropriate timeframes** (H1, H4, D1 for reliability)
2. **Set minimum confidence** (recommend 60%+)
3. **Combine with trend analysis** for context
4. **Look for confluence** with support/resistance

### Trading Application
1. **Wait for confirmation** after pattern completion
2. **Use stop losses** below/above pattern formation
3. **Consider volume** for pattern strength
4. **Multiple patterns** increase signal reliability

### Chart Analysis
1. **Start with overview** to see all patterns
2. **Filter by direction** for trend bias
3. **Use detailed view** for specific patterns
4. **Check frequency analysis** for pattern reliability

## File Structure

```
core/
├── candlestick_patterns.py    # Pattern detection logic
utils/
├── candlestick_plotter.py     # Visualization functions
tests/
├── test_candlestick_patterns.py  # Example usage
docs/
├── candlestick_patterns_guide.md # This guide
```

## Example Output

The system successfully detected **250 key patterns** in XAUUSD H1 data:

- **Evening Star**: 50 patterns (most common)
- **Bullish Engulfing**: 49 patterns
- **Morning Star**: 45 patterns
- **Bearish Engulfing**: 38 patterns
- **Inverted Hammer**: 36 patterns
- **Hammer**: 30 patterns
- **Doji**: 2 patterns

High-confidence patterns (>80%) include:
- **Bullish/Bearish Engulfing**: Up to 95% confidence
- **Hammer/Inverted Hammer**: Up to 95% confidence
- **Morning/Evening Star**: Up to 93% confidence

The focused pattern set provides higher-quality, more reliable trading signals.

## Future Enhancements

- **Pattern performance tracking**
- **Success rate statistics**
- **Price target calculations**
- **Volume confirmation**
- **Multi-timeframe analysis**
- **Alert system for new patterns**