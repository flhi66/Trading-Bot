#!/usr/bin/env python3
"""
EURUSD Multi-Timeframe Trading Strategy Demo with Backtesting

This script demonstrates the strategy using EURUSD data and generates
a comprehensive backtest report for a user-defined period.

MODIFICATION:
- Uses 1H data instead of 4H for trend analysis
- Removes chart plotting functionality
- Focuses on importing results to review trend analysis output
- Backtest period is now configurable (e.g., 180 days).
- Trend filter has been slightly relaxed for broader testing.
- The final report includes a detailed analysis of every trade, showing the
  market conditions at the time of entry for in-depth review.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trading_executor import MultiTimeframeTradingExecutor
from core.data_loader import load_and_resample
from core.backtester import BOSCHOCHBacktester, generate_backtest_report
from core.smart_money_concepts import MarketStructureAnalyzer
from core.structure_builder import build_market_structure
from utils.pattern_movements import detect_trend
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np

# This function remains unchanged from the previous version.
def analyze_trend_strength(data_1h: pd.DataFrame) -> tuple[str, float]:
    # ... (Implementation is identical to the previous version)
    if len(data_1h) < 20: return "sideways", 0.0
    from core.pivot_detector import find_pivots
    from core.trend_detector import detect_swing_points, detect_trend
    from core.structure_builder import get_market_analysis
    pivots = find_pivots(data_1h, prominence=0.001)
    swing_highs, swing_lows = detect_swing_points(data_1h, window=3)
    swing_trend = detect_trend(swing_highs, swing_lows)
    market_analysis = get_market_analysis(data_1h, prominence_factor=7.5, trend_window=4)
    structure_trend = market_analysis["trend"]
    recent_data = data_1h.tail(20)
    price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
    volatility = recent_data['High'].max() - recent_data['Low'].min()
    avg_price = recent_data['Close'].mean()
    volatility_pct = volatility / avg_price
    if len(pivots) >= 4:
        x_numeric = np.arange(len(pivots))
        z = np.polyfit(x_numeric, pivots['price'], 1)
        trend_slope = z[0]
        trend_strength_raw = abs(trend_slope) / avg_price
        trend_slope_score = min(trend_strength_raw * 1000, 1.0)
    else:
        trend_slope_score = 0.0
    highs = recent_data['High'].rolling(window=5).max()
    lows = recent_data['Low'].rolling(window=5).min()
    higher_highs = (highs.diff() > 0).sum()
    higher_lows = (lows.diff() > 0).sum()
    consistency_score = (higher_highs + higher_lows) / (2 * (len(highs) - 1)) if len(highs) > 1 else 0.0
    trend_agreement = 1.0 if swing_trend == structure_trend else (0.5 if swing_trend != "sideways" and structure_trend != "sideways" else 0.0)
    momentum_score = min(abs(price_change) / 0.02, 1.0)
    volatility_score = min(volatility_pct / 0.05, 1.0)
    trend_strength = (momentum_score * 0.25 + volatility_score * 0.20 + consistency_score * 0.20 + trend_slope_score * 0.20 + trend_agreement * 0.15)
    primary_trend = swing_trend
    if trend_strength < 0.4: return "sideways", trend_strength
    elif primary_trend == "uptrend" and price_change > 0.005: return "uptrend", trend_strength
    elif primary_trend == "downtrend" and price_change < -0.005: return "downtrend", trend_strength
    else: return "sideways", trend_strength


# This function remains unchanged from the previous version.
def enhanced_1m_confirmation(data_1m: pd.DataFrame, signal_direction: str, entry_time: pd.Timestamp, confirmation_window: int = 5) -> bool:
    # ... (Implementation is identical to the previous version)
    try:
        entry_index = data_1m.index.get_loc(entry_time)
    except KeyError:
        return False
    if entry_index + confirmation_window >= len(data_1m):
        return False
    confirmation_candles_df = data_1m.iloc[entry_index:entry_index + confirmation_window]
    confirming_candles_count = 0
    for _, candle in confirmation_candles_df.iterrows():
        if signal_direction == "BUY":
            if candle['Close'] > candle['Open']:
                confirming_candles_count += 1
                if candle['Close'] > candle['High'] * 0.998: confirming_candles_count += 0.5
        else:
            if candle['Close'] < candle['Open']:
                confirming_candles_count += 1
                if candle['Close'] < candle['Low'] * 1.002: confirming_candles_count += 0.5
    confirmation_score = confirming_candles_count / confirmation_window
    return confirmation_score >= 0.6

def should_trade_in_market_conditions(trend_direction: str, trend_strength: float, market_volatility: float) -> tuple[bool, list]:
    """
    Simplified market condition assessment using swing point analysis.
    - Focus on trend strength and direction
    - Use last week data for better trend identification
    - Simplified risk assessment for EURUSD
    
    Returns: (should_trade, detailed_conditions)
    """
    conditions = []
    
    # === 1. Trend Strength Check (Simplified) ===
    # TUNABLE PARAMETER: The required trend strength to consider a trade.
    # Lowering this value will allow trades in less trendy markets, potentially
    # increasing trade frequency but might also increase lower-quality signals.
    required_trend_strength = 0.3
    
    if trend_strength >= required_trend_strength:
        conditions.append(f"✅ Good trend detected (strength: {trend_strength:.2f})")
    elif trend_strength >= 0.3:
        conditions.append(f"⚠️ Moderate trend (strength: {trend_strength:.2f}) - proceed with caution")
        if trend_direction in ["uptrend", "downtrend"]:
            conditions.append("✅ Clear direction compensates for moderate strength")
        else:
            conditions.append("❌ Insufficient trend strength - avoid trading")
            return False, conditions
    else:
        conditions.append(f"❌ Weak trend (strength: {trend_strength:.2f}) - avoid trading")
        return False, conditions
    
    # === 2. Trend Direction Validation ===
    if trend_direction in ["uptrend", "downtrend"]:
        conditions.append(f"✅ Clear {trend_direction} direction confirmed")
    else:
        conditions.append("❌ Sideways market - avoid trading")
        return False, conditions
    
    # === 3. Simplified Volatility Analysis ===
    # TUNABLE PARAMETERS: The ideal volatility range.
    min_volatility, max_volatility = 0.005, 0.05
    if min_volatility <= market_volatility <= max_volatility:
        conditions.append(f"✅ Suitable volatility range ({market_volatility:.1%})")
    elif market_volatility < min_volatility:
        conditions.append(f"⚠️ Low volatility ({market_volatility:.1%}) - may limit profit potential")
        if trend_strength >= 0.6:
            conditions.append("✅ Strong trend compensates for low volatility")
        else:
            conditions.append("❌ Low volatility + weak trend - avoid trading")
            return False, conditions
    else:
        conditions.append(f"⚠️ High volatility ({market_volatility:.1%}) - increased risk")
        if market_volatility > 0.08:
            conditions.append("❌ Excessive volatility - avoid trading")
            return False, conditions
        elif trend_strength >= 0.7:
            conditions.append("✅ Strong trend can handle high volatility")
        else:
            conditions.append("❌ High volatility without strong trend - avoid trading")
            return False, conditions
    
    # === 4. Overall Assessment ===
    positive_conditions = len([c for c in conditions if "✅" in c])
    if positive_conditions >= 2:
        conditions.append("🎯 Market conditions suitable for trading")
        should_trade = True
    else:
        conditions.append("❌ Market conditions not suitable for trading")
        should_trade = False
    
    return should_trade, conditions

def enhanced_trend_detection(data_1h: pd.DataFrame, lookback_periods: list = [30, 60, 90]) -> tuple[str, float, dict]:
    """
    Enhanced trend detection that analyzes multiple timeframes to provide
    more consistent and reliable trend identification.
    
    Args:
        data_1h: 1H timeframe data
        lookback_periods: List of days to analyze (e.g., [30, 60, 90])
    
    Returns:
        (final_trend, confidence_score, detailed_analysis)
    """
    from core.pivot_detector import find_pivots
    
    print("\n🔍 ENHANCED MULTI-TIMEFRAME TREND ANALYSIS")
    print("=" * 60)
    
    if data_1h.empty or len(data_1h) < 30 * 24:  # Need at least 30 days of 1H data
        print("❌ Insufficient data for enhanced trend analysis")
        return "sideways", 0.0, {}
    
    trend_results = {}
    confidence_scores = {}
    
    for period_days in lookback_periods:
        candles_needed = period_days * 24
        if len(data_1h) >= candles_needed:
            period_data = data_1h.tail(candles_needed)
            
            # Find pivot points for this period
            pivots = find_pivots(period_data, prominence=0.001)
            
            if len(pivots) >= 3:
                # Use pattern_movements.py detect_trend
                from utils.pattern_movements import detect_trend
                trend = detect_trend(pivots)
                
                # Calculate additional trend strength metrics
                price_change = (period_data['Close'].iloc[-1] - period_data['Close'].iloc[0]) / period_data['Close'].iloc[0]
                price_change_pct = price_change * 100
                
                # Calculate trend consistency
                if len(pivots) >= 4:
                    x_numeric = np.arange(len(pivots))
                    z = np.polyfit(x_numeric, pivots['price'], 1)
                    trend_slope = z[0]
                    trend_strength = min(abs(trend_slope) / period_data['Close'].mean() * 1000, 1.0)
                else:
                    trend_strength = 0.5
                
                # Calculate volatility
                volatility = (period_data['High'].max() - period_data['Low'].min()) / period_data['Close'].mean()
                
                # Calculate momentum (recent vs overall)
                recent_data = period_data.tail(7 * 24)  # Last 7 days
                if len(recent_data) > 0:
                    recent_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
                    momentum_alignment = 1.0 if (trend == "Uptrend" and recent_change > 0) or (trend == "Downtrend" and recent_change < 0) else 0.5
                else:
                    momentum_alignment = 0.5
                
                # Overall confidence for this period
                period_confidence = (trend_strength * 0.4 + 
                                   min(abs(price_change) / 0.05, 1.0) * 0.3 + 
                                   momentum_alignment * 0.3)
                
                trend_results[period_days] = {
                    'trend': trend,
                    'price_change_pct': price_change_pct,
                    'trend_strength': trend_strength,
                    'volatility': volatility,
                    'momentum_alignment': momentum_alignment,
                    'confidence': period_confidence,
                    'pivot_count': len(pivots)
                }
                
                print(f"\n📊 {period_days}-Day Analysis:")
                print(f"   Trend: {trend}")
                print(f"   Price Change: {price_change_pct:+.2f}%")
                print(f"   Trend Strength: {trend_strength:.3f}")
                print(f"   Volatility: {volatility:.2%}")
                print(f"   Momentum Alignment: {momentum_alignment:.3f}")
                print(f"   Confidence: {period_confidence:.3f}")
                print(f"   Pivot Points: {len(pivots)}")
    
    # Determine final trend based on consensus
    if not trend_results:
        return "sideways", 0.0, {}
    
    # Count trend occurrences
    trend_counts = {}
    for period, result in trend_results.items():
        trend = result['trend'].lower()
        if trend not in trend_counts:
            trend_counts[trend] = 0
        trend_counts[trend] += 1
    
    # Find most common trend
    if trend_counts:
        most_common_trend = max(trend_counts, key=trend_counts.get)
        trend_consensus = trend_counts[most_common_trend] / len(trend_results)
    else:
        most_common_trend = "sideways"
        trend_consensus = 0.0
    
    # Calculate overall confidence
    avg_confidence = sum(result['confidence'] for result in trend_results.values()) / len(trend_results)
    
    # Adjust confidence based on consensus
    final_confidence = avg_confidence * (0.7 + 0.3 * trend_consensus)
    
    # Determine final trend with quality assessment
    if trend_consensus >= 0.67:  # At least 2/3 periods agree
        if most_common_trend == "uptrend":
            final_trend = "uptrend"
            trend_quality = "STRONG" if final_confidence >= 0.7 else "MODERATE"
        elif most_common_trend == "downtrend":
            final_trend = "downtrend"
            trend_quality = "STRONG" if final_confidence >= 0.7 else "MODERATE"
        else:
            final_trend = "sideways"
            trend_quality = "WEAK"
    else:
        final_trend = "sideways"
        trend_quality = "WEAK"
    
    print(f"\n🎯 FINAL TREND DETERMINATION")
    print("=" * 50)
    print(f"   Trend Consensus: {trend_consensus:.1%} ({most_common_trend.upper()})")
    print(f"   Final Trend: {final_trend.upper()}")
    print(f"   Trend Quality: {trend_quality}")
    print(f"   Confidence Level: {final_confidence:.1%}")
    
    return final_trend, final_confidence, trend_results

def display_detailed_trend_analysis(data_1h: pd.DataFrame) -> tuple[str, float]:
    """
    Simple trend analysis using the detect_trend function from pattern_movements.py:
    - Uses 90 days of 1H data for reliable swing detection
    - Imports trend detection logic from pattern_movements.py
    - No custom logic - just uses the existing function
    
    Returns: (trend_direction, trend_strength_score)
    """
    print("\n🔍 TREND ANALYSIS (Using pattern_movements.py with 1H data)")
    print("=" * 60)
    
    if data_1h.empty or len(data_1h) < 20:
        print("❌ Insufficient data for trend analysis")
        return "sideways", 0.0
    
    # Import required functions
    from core.pivot_detector import find_pivots
    
    # === FOCUS ON 90 DAYS OF 1H DATA ===
    # 90 days = 90 * 24 = 2160 1H candles (24 candles per day)
    # Use all available data up to 90 days
    days_to_analyze = min(90, len(data_1h) // 24)  # Convert candles to days
    candles_to_analyze = days_to_analyze * 24
    
    if len(data_1h) >= candles_to_analyze:
        trend_data = data_1h.tail(candles_to_analyze)
    else:
        trend_data = data_1h  # Use all available data
    
    print(f"📅 Data Period: Last {days_to_analyze} days ({len(trend_data)} 1H candles)")
    print(f"   From: {trend_data.index[0].strftime('%Y-%m-%d %H:%M')}")
    print(f"   To: {trend_data.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"   Timeframe: 1H (24 candles per day)")
    
    # === PIVOT POINT ANALYSIS ===
    print("\n📊 PIVOT POINT ANALYSIS")
    pivots = find_pivots(trend_data, prominence=0.001)
    print(f"   Pivot Points Found: {len(pivots)}")
    
    # === TREND DETECTION (Using pattern_movements.py) ===
    print("\n📊 TREND DETECTION (pattern_movements.py)")
    trend_direction = detect_trend(pivots)
    print(f"   Detected Trend: {trend_direction.upper()}")
    
    # === TREND STRENGTH CALCULATION ===
    print("\n📊 TREND STRENGTH CALCULATION")
    
    # Simple strength calculation based on pivot points and trend direction
    if trend_direction == "Sideways":
        trend_strength = 0.3
        print(f"   Base Strength: Weak (sideways market)")
    elif trend_direction in ["Uptrend", "Downtrend"]:
        if len(pivots) >= 3:
            trend_strength = 0.7
            print(f"   Base Strength: Strong (clear direction with multiple pivot points)")
        else:
            trend_strength = 0.5
            print(f"   Base Strength: Moderate (clear direction)")
    else:
        trend_strength = 0.4
        print(f"   Base Strength: Weak (neutral market)")
    
    # Calculate price change for additional strength
    price_change = (trend_data['Close'].iloc[-1] - trend_data['Close'].iloc[0]) / trend_data['Close'].iloc[0]
    price_change_pct = price_change * 100
    print(f"   Period Price Change: {price_change_pct:+.2f}%")
    
    # Adjust strength based on price movement
    if abs(price_change) > 0.02:  # 2% move over 90 days
        trend_strength += 0.2
        print(f"   Price Movement Bonus: +0.2 (strong period move)")
    elif abs(price_change) > 0.01:  # 1% move over 90 days
        trend_strength += 0.1
        print(f"   Price Movement Bonus: +0.1 (moderate period move)")
    else:
        print(f"   Price Movement Bonus: +0.0 (weak period move)")
    
    # Cap strength at 1.0
    trend_strength = min(trend_strength, 1.0)
    
    print(f"   ─────────────────────────────")
    print(f"   FINAL TREND STRENGTH: {trend_strength:.3f}")
    
    # === FINAL TREND DETERMINATION ===
    print("\n🎯 FINAL TREND DETERMINATION")
    print("=" * 50)
    
    # Use the trend from pattern_movements.py
    primary_trend = trend_direction.lower()
    
    # Apply strength threshold
    if trend_strength < 0.4:
        final_trend = "sideways"
        trend_quality = "WEAK"
    elif primary_trend == "uptrend" and price_change > 0.008:  # 0.8% positive move
        final_trend = "uptrend"
        trend_quality = "STRONG" if trend_strength >= 0.7 else "MODERATE"
    elif primary_trend == "downtrend" and price_change < -0.008:  # 0.8% negative move
        final_trend = "downtrend"
        trend_quality = "STRONG" if trend_strength >= 0.7 else "MODERATE"
    else:
        final_trend = "sideways"
        trend_quality = "WEAK"
    
    print(f"   Primary Trend Method: pattern_movements.py detect_trend")
    print(f"   Primary Trend: {primary_trend.upper()}")
    print(f"   Final Trend: {final_trend.upper()}")
    print(f"   Trend Quality: {trend_quality}")
    print(f"   Confidence Level: {trend_strength:.1%}")
    
    return final_trend, trend_strength

# This function remains unchanged from the previous version.
def demo_eurusd_strategy():
    # ... (Implementation is identical to the previous version)
    print("🎯 EURUSD Multi-Timeframe Trading Strategy Demo\n" + "=" * 60)
    executor = MultiTimeframeTradingExecutor(symbol="EURUSD", risk_per_trade=0.01, stop_loss_pips=50.0, risk_reward_ratio=1.5, confidence_threshold=0.6, pip_value=0.0001)
    data_file = "data/EURUSD_M1.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}"); return None, executor
    print(f"\n📈 Running strategy on {data_file} for CURRENT market conditions")
    # ENHANCED: Load data for trend strength analysis
    print("🔍 Loading data for enhanced analysis...")
    resampled_data = load_and_resample(data_file, days_back=60)
    data_1h = resampled_data.get('1H')  # Use 1H data for trend analysis
    
    if data_1h is not None and not data_1h.empty:
        # Use trend analysis with 1H data (90 days focus) using pattern_movements.py
        trend_direction, trend_strength = display_detailed_trend_analysis(data_1h)
        
        # Calculate market volatility from 1H data
        recent_data = data_1h.tail(90)  # Last 90 days of 1H data
        volatility = (recent_data['High'].max() - recent_data['Low'].min()) / recent_data['Close'].mean()
        
        print(f"\n📊 Enhanced Market Analysis Summary:")
        print(f"   1H Trend (90 days): {trend_direction.upper()}")
        print(f"   Trend Strength: {trend_strength:.2f}")
        print(f"   Market Volatility: {volatility:.1%}")
        
        # Check if market conditions are suitable for trading
        should_trade, conditions = should_trade_in_market_conditions(trend_direction, trend_strength, volatility)
        
        print(f"\n🎯 Market Condition Assessment:")
        for condition in conditions:
            print(f"   {condition}")
        
        if not should_trade:
            print(f"\n⚠️  Market conditions not suitable for trading")
            print(f"   Strategy execution skipped due to poor market conditions")
            return None, executor
        else:
            print(f"\n✅ Market conditions suitable for trading - proceeding with strategy")
    results = executor.run_strategy(data_file, days_back=60)
    print(f"\n📊 Strategy Results:\n   Signals Generated: {results['signals_generated']}\n   Trades Executed: {results['trades_executed']}\n   Total P&L: ${results['total_pnl']:.2f}")
    return results, executor

def run_eurusd_backtest(data_15m: pd.DataFrame, data_1h: pd.DataFrame):
    """Run comprehensive backtest on pre-loaded EURUSD data."""
    print("\n🚀 EURUSD Comprehensive Backtest")
    print("=" * 60)
    
    if data_15m is None or data_15m.empty:
        print("❌ No 15M data available")
        return None
    
    print(f"✅ Using {len(data_15m)} 15M candles and {len(data_1h)} 1H candles for backtest.")
    
    market_analyzer = MarketStructureAnalyzer()
    print("🔍 Analyzing market structure to find all potential events...")
    structure = build_market_structure(data_15m)
    events = market_analyzer.get_market_events(structure)
    print(f"✅ Found {len(events)} potential market events")
    
    a_plus_events = []
    print("⏳ Filtering events based on ENHANCED 1H trend conditions (no lookahead bias)...")
    
    if data_1h is not None and not data_1h.empty:
        # Use enhanced trend detection for overall market assessment
        overall_trend, overall_confidence, trend_details = enhanced_trend_detection(data_1h)
        
        print(f"\n📊 OVERALL MARKET ASSESSMENT:")
        print(f"   Primary Trend: {overall_trend.upper()}")
        print(f"   Overall Confidence: {overall_confidence:.3f}")
        
        for event in events:
            # For each event, analyze the trend in the 1H data *prior* to the event
            event_time = event.timestamp
            
            # Get data prior to the event for trend analysis
            prior_1h_data = data_1h.loc[data_1h.index < event_time]
            
            if len(prior_1h_data) < 30 * 24:  # Need at least 30 days of data
                continue
            
            # Use enhanced trend detection for this specific event
            event_trend, event_confidence, event_trend_details = enhanced_trend_detection(prior_1h_data)
            
            # Calculate volatility for the 1H data prior to the event time
            prior_1h_recent = prior_1h_data.tail(90)  # Last 90 days
            if len(prior_1h_recent) < 20: 
                continue
            
            volatility = (prior_1h_recent['High'].max() - prior_1h_recent['Low'].min()) / prior_1h_recent['Close'].mean()
            
            # More flexible trend alignment check
            should_trade, conditions = should_trade_in_market_conditions(event_trend, event_confidence, volatility)
            
            if should_trade and event.confidence >= 0.5:
                # Enhanced trend alignment with multiple criteria
                trend_alignment_score = 0
                
                # 1. Check if event direction aligns with event-specific trend
                if (event.direction in ["BUY", "Bullish"] and event_trend == "uptrend") or (event.direction in ["SELL", "Bearish"] and event_trend == "downtrend"):
                    trend_alignment_score += 1
                    print(f"   ✅ Direction-Trend Alignment: {event.direction} aligns with {event_trend}")
                else:
                    print(f"   ❌ Direction-Trend Mismatch: {event.direction} vs {event_trend}")
                
                # 2. Check if event direction aligns with overall market trend
                if (event.direction in ["BUY", "Bullish"] and overall_trend == "uptrend") or (event.direction in ["SELL", "Bearish"] and overall_trend == "downtrend"):
                    trend_alignment_score += 1
                    print(f"   ✅ Overall Market Alignment: {event.direction} aligns with overall {overall_trend}")
                else:
                    print(f"   ❌ Overall Market Mismatch: {event.direction} vs overall {overall_trend}")
                
                # 3. Check if event confidence is high enough
                if event.confidence >= 0.7:
                    trend_alignment_score += 1
                    print(f"   ✅ High Event Confidence: {event.confidence:.3f}")
                else:
                    print(f"   ⚠️ Moderate Event Confidence: {event.confidence:.3f}")
                
                print(f"   📊 Final Alignment Score: {trend_alignment_score}/3")
                
                # Accept trades with at least 2 out of 3 alignment criteria
                if trend_alignment_score >= 2:
                    a_plus_events.append(event)
                    print(f"✅ Accepted {event.direction} signal at {event_time.strftime('%Y-%m-%d %H:%M')} "
                          f"(Event Trend: {event_trend}, Overall Trend: {overall_trend}, Alignment Score: {trend_alignment_score}/3)")
                else:
                    print(f"❌ Rejected {event.direction} signal at {event_time.strftime('%Y-%m-%d %H:%M')} "
                          f"(Event Trend: {event_trend}, Overall Trend: {overall_trend}, Alignment Score: {trend_alignment_score}/3)")
        
        print(f"\n✅ Filtered down to {len(a_plus_events)} events meeting enhanced trend conditions.")
        
        if not a_plus_events:
            print("⚠️ No events met the enhanced trend conditions for trading.")
            print("🔍 This could indicate:")
            print("   - Mixed or unclear trend signals")
            print("   - Low confidence in trend direction")
            print("   - High volatility making trend identification difficult")
            print("   - Sideways market conditions")
            
            # Provide alternative analysis
            print("\n📊 ALTERNATIVE ANALYSIS:")
            print("   Consider relaxing trend requirements or using different timeframes")
            print("   Current overall trend: " + overall_trend.upper())
            print("   Overall confidence: " + f"{overall_confidence:.1%}")
            
            return None
    else:
        print("⚠️ 1H data not available. Using original confidence-based filtering.")
        a_plus_events = [e for e in events if e.confidence >= 0.5]

    backtester = BOSCHOCHBacktester(
        initial_capital=10000, risk_per_trade=0.01, reward_risk_ratio=1.5,
        max_trade_duration=72, confidence_threshold=0.5, stop_loss_atr_multiplier=3.0
    )
    print(f"\n🚀 Running backtest on {len(a_plus_events)} filtered A+ events...")
    backtest_result = backtester.run_backtest(data_15m, a_plus_events)
    
    return backtest_result

def generate_eurusd_report(backtest_result, data_1h: pd.DataFrame):
    """
    Generate comprehensive EURUSD backtest report, including
    detailed analysis of each trade's entry conditions.
    """
    print("\n📊 Generating EURUSD Backtest Report")
    print("=" * 60)
    
    if backtest_result is None:
        print("❌ No backtest results to report")
        return
    
    output_dir = "Generated Reports/generated_eurusd_backtest"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate report without charts - just import results
        generate_backtest_report(backtest_result, output_dir)
        print(f"✅ EURUSD backtest report generated in '{output_dir}' directory")
    except Exception as e:
        print(f"⚠️ Warning: Could not generate full report: {str(e)}")
    
    print(f"\n📈 Key Performance Metrics:")
    print(f"   Total Trades: {backtest_result.total_trades}")
    print(f"   Win Rate: {backtest_result.win_rate:.1%}")
    print(f"   Total P&L: ${backtest_result.total_pnl:.2f} ({backtest_result.total_pnl_pct:.2f}%)")
    print(f"   Max Drawdown: ${backtest_result.max_drawdown:.2f} ({backtest_result.max_drawdown_pct:.2f}%)")
    print(f"   Profit Factor: {backtest_result.profit_factor:.2f}")
    print(f"   Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")

    print("\n" + "="*20 + " Detailed Trade-by-Trade Analysis " + "="*20)
    if not backtest_result.trades:
        print("\n   No trades were executed in this backtest period.")
    else:
        for i, trade in enumerate(backtest_result.trades, 1):
            print(f"\n--- Trade #{i}: {trade.direction} ({trade.entry_type}) ---")
            result_status = "WIN" if trade.pnl > 0 else "LOSS"
            print(f"   Result: {result_status} | P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
            print(f"   Entry: {trade.entry_time.strftime('%Y-%m-%d %H:%M')} @ {trade.entry_price:.5f}")
            print(f"   Exit:  {trade.exit_time.strftime('%Y-%m-%d %H:%M')} @ {trade.exit_price:.5f}")
            print(f"   Duration: {trade.duration} | Exit Reason: {trade.exit_reason}")

            # Re-analyze and display the market conditions at the moment of entry
            if data_1h is not None:
                trend_direction, trend_strength = display_detailed_trend_analysis(
                    data_1h.loc[data_1h.index < trade.entry_time]
                )
                prior_1h_data = data_1h.loc[data_1h.index < trade.entry_time].tail(90)
                if not prior_1h_data.empty:
                    volatility = (prior_1h_data['High'].max() - prior_1h_data['Low'].min()) / prior_1h_data['Close'].mean()
                else:
                    volatility = 0

                print(f"   --- Market Conditions at Entry ---")
                print(f"   1H Trend Direction: {trend_direction.upper()}")
                print(f"   Trend Strength:     {trend_strength:.3f}")
                print(f"   Market Volatility:  {volatility:.2%}")

def main():
    """Main EURUSD demonstration function"""
    # --- CONFIGURATION ---
    BACKTEST_DAYS = 180  # Set backtest duration (e.g., 180 for 6 months, 365 for a year)

    print("🚀 EURUSD Multi-Timeframe Trading Strategy")
    print("=" * 70)
    print(f"📅 Period: Last {BACKTEST_DAYS} days")
    print("🎯 Strategy: Multi-timeframe with BOS/CHOCH A+ entries and dynamic trend filtering.")
    print("🔍 TREND DETECTION: Using pattern_movements.py detect_trend function with 1H data")
    print("📊 TIMEFRAME: 1H for trend analysis (90 days focus) + 15M for entries + 1M for confirmation")
    
    try:
        # --- Data Loading ---
        data_file = "data/EURUSD_M1.csv"
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return

        print(f"\n📈 Loading data for the last {BACKTEST_DAYS} days...")
        resampled_data = load_and_resample(data_file, days_back=BACKTEST_DAYS)
        data_1h = resampled_data.get('1H')  # Use 1H data for trend analysis
        data_15m = resampled_data.get('15M')
        
        if data_15m is None or data_1h is None:
            print("❌ Required data not available for backtest")
            return
        
        # --- Run Comprehensive Backtest ---
        backtest_results = run_eurusd_backtest(data_15m, data_1h)
        
        # --- Generate Comprehensive Report with Detailed Analysis ---
        if backtest_results:
            generate_eurusd_report(backtest_results, data_1h)
        
        print("\n🎉 EURUSD analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ An error occurred during the process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()