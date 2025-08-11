#!/usr/bin/env python3
"""
Comprehensive BOS/CHOCH Detection Verification Script
Analyzes the quality of BOS/CHOCH detection and identifies areas for improvement
"""

from core.data_loader import load_and_resample
from core.structure_builder import get_market_analysis
from core.smart_money_concepts import MarketStructureAnalyzer, StructurePoint, SwingType
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def analyze_structure_quality(structure: List[Dict]) -> Dict:
    """Analyze the quality of market structure points"""
    if not structure:
        return {"error": "No structure data"}
    
    analysis = {
        "total_points": len(structure),
        "swing_distribution": {},
        "price_ranges": {},
        "time_gaps": [],
        "potential_issues": []
    }
    
    # Analyze swing distribution
    for point in structure:
        swing_type = point['type']
        if swing_type not in analysis['swing_distribution']:
            analysis['swing_distribution'][swing_type] = 0
        analysis['swing_distribution'][swing_type] += 1
    
    # Analyze price ranges
    prices = [p['price'] for p in structure]
    analysis['price_ranges'] = {
        'min': min(prices),
        'max': max(prices),
        'range': max(prices) - min(prices),
        'avg_gap': np.mean([abs(prices[i] - prices[i-1]) for i in range(1, len(prices))])
    }
    
    # Analyze time gaps
    timestamps = [pd.Timestamp(p['timestamp']) for p in structure]
    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # hours
        analysis['time_gaps'].append(gap)
    
    # Identify potential issues
    if len(structure) < 10:
        analysis['potential_issues'].append("Very few structure points - may miss important swings")
    
    if analysis['swing_distribution'].get('HH', 0) == 0 or analysis['swing_distribution'].get('LL', 0) == 0:
        analysis['potential_issues'].append("Missing either HH or LL swings - incomplete structure")
    
    # Check for unrealistic price gaps
    if analysis['price_ranges']['avg_gap'] > 100:  # More than 100 points average gap
        analysis['potential_issues'].append("Large average price gaps - may be missing intermediate swings")
    
    return analysis

def verify_bos_detection(structure: List[Dict], events: List) -> Dict:
    """Verify BOS detection quality"""
    bos_events = [e for e in events if hasattr(e, 'event_type') and e.event_type.value == 'BOS']
    
    verification = {
        "total_bos": len(bos_events),
        "bullish_bos": len([e for e in bos_events if e.direction == "Bullish"]),
        "bearish_bos": len([e for e in bos_events if e.direction == "Bearish"]),
        "verification_results": [],
        "issues": []
    }
    
    for event in bos_events:
        # Verify BOS logic
        if event.direction == "Bullish":
            # Should be HH breaking previous HH
            current_price = event.price
            broken_level_price = event.broken_level['price']
            
            if current_price <= broken_level_price:
                verification['issues'].append(f"Bullish BOS @ {current_price} should be above broken level {broken_level_price}")
            else:
                verification['verification_results'].append(f"✅ Bullish BOS @ {current_price} correctly above {broken_level_price}")
                
        elif event.direction == "Bearish":
            # Should be LL breaking previous LL
            current_price = event.price
            broken_level_price = event.broken_level['price']
            
            if current_price >= broken_level_price:
                verification['issues'].append(f"Bearish BOS @ {current_price} should be below broken level {broken_level_level}")
            else:
                verification['verification_results'].append(f"✅ Bearish BOS @ {current_price} correctly below {broken_level_price}")
    
    return verification

def verify_choch_detection(structure: List[Dict], events: List) -> Dict:
    """Verify CHOCH detection quality"""
    choch_events = [e for e in events if hasattr(e, 'event_type') and e.event_type.value == 'CHOCH']
    
    verification = {
        "total_choch": len(choch_events),
        "bullish_choch": len([e for e in choch_events if e.direction == "Bullish"]),
        "bearish_choch": len([e for e in choch_events if e.direction == "Bearish"]),
        "verification_results": [],
        "issues": []
    }
    
    for event in choch_events:
        # Verify CHOCH logic
        if event.direction == "Bearish":
            # Should be lower swing breaking uptrend support
            current_price = event.price
            broken_level_price = event.broken_level['price']
            
            if current_price >= broken_level_price:
                verification['issues'].append(f"Bearish CHOCH @ {current_price} should be below support {broken_level_price}")
            else:
                verification['verification_results'].append(f"✅ Bearish CHOCH @ {current_price} correctly below support {broken_level_price}")
                
        elif event.direction == "Bullish":
            # Should be higher swing breaking downtrend resistance
            current_price = event.price
            broken_level_price = event.broken_level['price']
            
            if current_price <= broken_level_price:
                verification['issues'].append(f"Bullish CHOCH @ {current_price} should be above resistance {broken_level_price}")
            else:
                verification['verification_results'].append(f"✅ Bullish CHOCH @ {current_price} correctly above resistance {broken_level_price}")
    
    return verification

def analyze_pattern_quality(structure: List[Dict]) -> Dict:
    """Analyze the quality of swing patterns"""
    if len(structure) < 4:
        return {"error": "Insufficient structure points"}
    
    analysis = {
        "pattern_sequences": [],
        "trend_consistency": [],
        "potential_improvements": []
    }
    
    # Analyze pattern sequences
    for i in range(2, len(structure)):
        current = structure[i]
        prev1 = structure[i-1]
        prev2 = structure[i-2]
        
        # Check for valid BOS patterns
        if current['type'] == 'HH' and prev1['type'] == 'HL' and prev2['type'] == 'HH':
            if current['price'] > prev2['price']:
                analysis['pattern_sequences'].append(f"✅ Valid Bullish BOS: {prev2['type']}@{prev2['price']:.2f} -> {prev1['type']}@{prev1['price']:.2f} -> {current['type']}@{current['price']:.2f}")
            else:
                analysis['pattern_sequences'].append(f"❌ Invalid Bullish BOS: {current['price']:.2f} not above {prev2['price']:.2f}")
        
        elif current['type'] == 'LL' and prev1['type'] == 'LH' and prev2['type'] == 'LL':
            if current['price'] < prev2['price']:
                analysis['pattern_sequences'].append(f"✅ Valid Bearish BOS: {prev2['type']}@{prev2['price']:.2f} -> {prev1['type']}@{prev1['price']:.2f} -> {current['type']}@{current['price']:.2f}")
            else:
                analysis['pattern_sequences'].append(f"❌ Invalid Bearish BOS: {current['price']:.2f} not below {prev2['price']:.2f}")
    
    # Check trend consistency
    uptrend_count = 0
    downtrend_count = 0
    
    for i in range(3, len(structure)):
        recent = structure[i-3:i+1]
        hh_count = sum(1 for p in recent if p['type'] == 'HH')
        ll_count = sum(1 for p in recent if p['type'] == 'LL')
        
        if hh_count > ll_count:
            uptrend_count += 1
        elif ll_count > hh_count:
            downtrend_count += 1
    
    analysis['trend_consistency'] = {
        'uptrend_periods': uptrend_count,
        'downtrend_periods': downtrend_count,
        'consistency_score': max(uptrend_count, downtrend_count) / (uptrend_count + downtrend_count) if (uptrend_count + downtrend_count) > 0 else 0
    }
    
    return analysis

def main():
    print("🔍 COMPREHENSIVE BOS/CHOCH DETECTION VERIFICATION")
    print("=" * 60)
    
    # Load data
    symbol = "XAUUSD_H1.csv"
    resampled = load_and_resample(f"data/{symbol}", days_back=60)
    h1_data = resampled.get("1H")
    
    if h1_data is None or h1_data.empty:
        print(f"❌ ERROR: No data loaded for the '1H' timeframe.")
        return
    
    print(f"📊 Loaded {len(h1_data)} H1 candles from {h1_data.index.min()} to {h1_data.index.max()}")
    
    # Get market structure
    analysis = get_market_analysis(h1_data, prominence_factor=2.5)
    structure = analysis['structure']
    
    print(f"\n📊 MARKET STRUCTURE ANALYSIS")
    print(f"Total structure points: {len(structure)}")
    
    # Analyze structure quality
    structure_quality = analyze_structure_quality(structure)
    print(f"\n🔍 Structure Quality Analysis:")
    print(f"  Swing Distribution: {structure_quality['swing_distribution']}")
    print(f"  Price Range: {structure_quality['price_ranges']['min']:.2f} - {structure_quality['price_ranges']['max']:.2f}")
    print(f"  Average Price Gap: {structure_quality['price_ranges']['avg_gap']:.2f} points")
    
    if structure_quality['potential_issues']:
        print(f"  ⚠️  Potential Issues:")
        for issue in structure_quality['potential_issues']:
            print(f"    - {issue}")
    
    # Get market events
    analyzer = MarketStructureAnalyzer(confidence_threshold=0.5)
    all_events = analyzer.get_market_events(structure)
    
    print(f"\n📊 EVENT DETECTION SUMMARY")
    print(f"Total events detected: {len(all_events)}")
    
    # Verify BOS detection
    bos_verification = verify_bos_detection(structure, all_events)
    print(f"\n🔍 BOS Detection Verification:")
    print(f"  Total BOS: {bos_verification['total_bos']}")
    print(f"  Bullish: {bos_verification['bullish_bos']}, Bearish: {bos_verification['bearish_bos']}")
    
    if bos_verification['issues']:
        print(f"  ❌ Issues found:")
        for issue in bos_verification['issues']:
            print(f"    - {issue}")
    else:
        print(f"  ✅ All BOS events verified correctly")
    
    # Verify CHOCH detection
    choch_verification = verify_choch_detection(structure, all_events)
    print(f"\n🔍 CHOCH Detection Verification:")
    print(f"  Total CHOCH: {choch_verification['total_choch']}")
    print(f"  Bullish: {choch_verification['bullish_choch']}, Bearish: {choch_verification['bearish_choch']}")
    
    if choch_verification['issues']:
        print(f"  ❌ Issues found:")
        for issue in choch_verification['issues']:
            print(f"    - {issue}")
    else:
        print(f"  ✅ All CHOCH events verified correctly")
    
    # Analyze pattern quality
    pattern_quality = analyze_pattern_quality(structure)
    print(f"\n🔍 Pattern Quality Analysis:")
    
    if 'error' not in pattern_quality:
        print(f"  Trend Consistency Score: {pattern_quality['trend_consistency']['consistency_score']:.2f}")
        print(f"  Uptrend Periods: {pattern_quality['trend_consistency']['uptrend_periods']}")
        print(f"  Downtrend Periods: {pattern_quality['trend_consistency']['downtrend_periods']}")
        
        if pattern_quality['pattern_sequences']:
            print(f"  Pattern Sequences (showing last 5):")
            for pattern in pattern_quality['pattern_sequences'][-5:]:
                print(f"    {pattern}")
    
    # Overall assessment
    print(f"\n📊 OVERALL ASSESSMENT")
    print("=" * 60)
    
    total_issues = len(bos_verification['issues']) + len(choch_verification['issues'])
    
    if total_issues == 0:
        print("✅ EXCELLENT: All BOS/CHOCH detections are verified correctly!")
        print("   The detection logic is working properly.")
    elif total_issues <= 2:
        print("🟡 GOOD: Most detections are correct with minor issues.")
        print("   Consider addressing the identified issues for improvement.")
    else:
        print("🔴 NEEDS IMPROVEMENT: Multiple detection issues found.")
        print("   Review the detection logic and validation criteria.")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if structure_quality['potential_issues']:
        print("  1. Address structure quality issues first:")
        for issue in structure_quality['potential_issues']:
            print(f"     - {issue}")
    
    if bos_verification['issues']:
        print("  2. Review BOS detection logic:")
        for issue in bos_verification['issues']:
            print(f"     - {issue}")
    
    if choch_verification['issues']:
        print("  3. Review CHOCH detection logic:")
        for issue in choch_verification['issues']:
            print(f"     - {issue}")
    
    if not structure_quality['potential_issues'] and total_issues == 0:
        print("  1. ✅ Current implementation is working well")
        print("  2. Consider adding more validation rules for edge cases")
        print("  3. Monitor performance with different market conditions")
    
    print(f"\n🎯 Next Steps:")
    print("  - Review the detailed verification results above")
    print("  - Address any identified issues")
    print("  - Test with different timeframes and instruments")
    print("  - Consider adding more sophisticated validation rules")

if __name__ == "__main__":
    main()
