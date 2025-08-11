#!/usr/bin/env python3
"""
Detailed BOS/CHOCH Pattern Analysis
Examines specific patterns and identifies areas for improvement
"""

from core.data_loader import load_and_resample
from core.structure_builder import get_market_analysis
from core.smart_money_concepts import MarketStructureAnalyzer, StructurePoint, SwingType
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def analyze_event_timing(events: List) -> Dict:
    """Analyze the timing and frequency of BOS/CHOCH events"""
    if not events:
        return {"error": "No events to analyze"}
    
    analysis = {
        "event_frequency": {},
        "time_between_events": [],
        "consecutive_events": [],
        "potential_issues": []
    }
    
    # Count event types
    for event in events:
        event_type = event.event_type.value
        if event_type not in analysis['event_frequency']:
            analysis['event_frequency'][event_type] = 0
        analysis['event_frequency'][event_type] += 1
    
    # Analyze time between events
    sorted_events = sorted(events, key=lambda x: x.timestamp)
    for i in range(1, len(sorted_events)):
        time_diff = (sorted_events[i].timestamp - sorted_events[i-1].timestamp).total_seconds() / 3600
        analysis['time_between_events'].append(time_diff)
    
    # Check for consecutive events of same type
    consecutive_count = 1
    for i in range(1, len(sorted_events)):
        if (sorted_events[i].event_type == sorted_events[i-1].event_type and 
            sorted_events[i].direction == sorted_events[i-1].direction):
            consecutive_count += 1
        else:
            if consecutive_count > 1:
                analysis['consecutive_events'].append({
                    'type': sorted_events[i-1].event_type.value,
                    'direction': sorted_events[i-1].direction,
                    'count': consecutive_count
                })
            consecutive_count = 1
    
    # Check for potential issues
    if len(analysis['time_between_events']) > 0:
        avg_time = np.mean(analysis['time_between_events'])
        if avg_time < 6:  # Less than 6 hours between events
            analysis['potential_issues'].append(f"Very frequent events (avg {avg_time:.1f}h apart) - may be over-detecting")
        
        if max(analysis['time_between_events']) > 168:  # More than 1 week
            analysis['potential_issues'].append("Very long gaps between events - may be missing important moves")
    
    # Check for consecutive events
    for consecutive in analysis['consecutive_events']:
        if consecutive['count'] > 2:
            analysis['potential_issues'].append(f"Multiple consecutive {consecutive['type']} events ({consecutive['count']}) - may need consolidation logic")
    
    return analysis

def analyze_price_levels(events: List) -> Dict:
    """Analyze the price levels and clustering of BOS/CHOCH events"""
    if not events:
        return {"error": "No events to analyze"}
    
    analysis = {
        "price_clusters": [],
        "level_effectiveness": {},
        "potential_issues": []
    }
    
    # Group events by price proximity (within 20 points)
    price_threshold = 20
    processed_events = set()
    
    for i, event1 in enumerate(events):
        if i in processed_events:
            continue
            
        cluster = [event1]
        processed_events.add(i)
        
        for j, event2 in enumerate(events[i+1:], i+1):
            if j in processed_events:
                continue
                
            if abs(event1.price - event2.price) <= price_threshold:
                cluster.append(event2)
                processed_events.add(j)
        
        if len(cluster) > 1:
            analysis['price_clusters'].append({
                'price_range': (min(e.price for e in cluster), max(e.price for e in cluster)),
                'events': cluster,
                'count': len(cluster)
            })
    
    # Analyze level effectiveness
    for event in events:
        level_name = event.broken_level['name']
        if level_name not in analysis['level_effectiveness']:
            analysis['level_effectiveness'][level_name] = {'count': 0, 'prices': []}
        
        analysis['level_effectiveness'][level_name]['count'] += 1
        analysis['level_effectiveness'][level_name]['prices'].append(event.broken_level['price'])
    
    # Check for potential issues
    if analysis['price_clusters']:
        large_clusters = [c for c in analysis['price_clusters'] if c['count'] > 2]
        if large_clusters:
            analysis['potential_issues'].append(f"Large price clusters detected - {len(large_clusters)} clusters with >2 events")
    
    # Check for level concentration
    for level_name, data in analysis['level_effectiveness'].items():
        if data['count'] > 3:
            analysis['potential_issues'].append(f"Level {level_name} broken {data['count']} times - may need stronger validation")
    
    return analysis

def analyze_trend_consistency(events: List) -> Dict:
    """Analyze the consistency of trend changes indicated by CHOCH events"""
    if not events:
        return {"error": "No events to analyze"}
    
    analysis = {
        "trend_changes": [],
        "trend_reversals": [],
        "consistency_score": 0,
        "potential_issues": []
    }
    
    # Track trend changes
    current_trend = None
    trend_changes = []
    
    for event in events:
        if event.event_type.value == 'CHOCH':
            if event.direction == 'Bullish':
                if current_trend == 'downtrend':
                    trend_changes.append(('downtrend', 'uptrend', event))
                current_trend = 'uptrend'
            elif event.direction == 'Bearish':
                if current_trend == 'uptrend':
                    trend_changes.append(('uptrend', 'downtrend', event))
                current_trend = 'downtrend'
    
    analysis['trend_changes'] = trend_changes
    
    # Check for trend reversals (CHOCH followed by opposite CHOCH)
    for i in range(len(events) - 1):
        if (events[i].event_type.value == 'CHOCH' and 
            events[i+1].event_type.value == 'CHOCH' and
            events[i].direction != events[i+1].direction):
            analysis['trend_reversals'].append({
                'first': events[i],
                'second': events[i+1],
                'time_gap': (events[i+1].timestamp - events[i].timestamp).total_seconds() / 3600
            })
    
    # Calculate consistency score
    if trend_changes:
        # Check if trend changes are logical (not too frequent)
        avg_time_between_changes = np.mean([
            (trend_changes[i+1][2].timestamp - trend_changes[i][2].timestamp).total_seconds() / 3600
            for i in range(len(trend_changes) - 1)
        ]) if len(trend_changes) > 1 else 0
        
        if avg_time_between_changes > 24:  # More than 1 day between trend changes
            analysis['consistency_score'] = 0.8
        elif avg_time_between_changes > 12:  # More than 12 hours
            analysis['consistency_score'] = 0.6
        else:
            analysis['consistency_score'] = 0.4
    else:
        analysis['consistency_score'] = 1.0  # No trend changes = consistent
    
    # Check for potential issues
    if analysis['trend_reversals']:
        quick_reversals = [r for r in analysis['trend_reversals'] if r['time_gap'] < 12]
        if quick_reversals:
            analysis['potential_issues'].append(f"Quick trend reversals detected - {len(quick_reversals)} reversals within 12h")
    
    if len(trend_changes) > 5:
        analysis['potential_issues'].append("Many trend changes detected - may be over-sensitive to market noise")
    
    return analysis

def analyze_confidence_distribution(events: List) -> Dict:
    """Analyze the confidence distribution of detected events"""
    if not events:
        return {"error": "No events to analyze"}
    
    analysis = {
        "confidence_ranges": {},
        "low_confidence_events": [],
        "confidence_issues": []
    }
    
    # Group events by confidence ranges
    for event in events:
        confidence = event.confidence
        
        if confidence >= 0.9:
            range_key = "0.9-1.0"
        elif confidence >= 0.8:
            range_key = "0.8-0.9"
        elif confidence >= 0.7:
            range_key = "0.7-0.8"
        elif confidence >= 0.6:
            range_key = "0.6-0.7"
        else:
            range_key = "0.5-0.6"
        
        if range_key not in analysis['confidence_ranges']:
            analysis['confidence_ranges'][range_key] = 0
        analysis['confidence_ranges'][range_key] += 1
        
        # Flag low confidence events
        if confidence < 0.7:
            analysis['low_confidence_events'].append({
                'event': event,
                'confidence': confidence,
                'reason': 'Below 70% confidence threshold'
            })
    
    # Check for confidence issues
    if len(analysis['low_confidence_events']) > len(events) * 0.3:
        analysis['confidence_issues'].append("High percentage of low confidence events - may need threshold adjustment")
    
    if not analysis['confidence_ranges'].get('0.9-1.0', 0):
        analysis['confidence_issues'].append("No high confidence events - may need validation rule improvements")
    
    return analysis

def main():
    print("🔍 DETAILED BOS/CHOCH PATTERN ANALYSIS")
    print("=" * 70)
    
    # Load data
    symbol = "XAUUSD_H1.csv"
    resampled = load_and_resample(f"data/{symbol}", days_back=60)
    h1_data = resampled.get("1H")
    
    if h1_data is None or h1_data.empty:
        print(f"❌ ERROR: No data loaded for the '1H' timeframe.")
        return
    
    print(f"📊 Loaded {len(h1_data)} H1 candles from {h1_data.index.min()} to {h1_data.index.max()}")
    
    # Get market structure and events
    analysis = get_market_analysis(h1_data, prominence_factor=2.5)
    structure = analysis['structure']
    
    analyzer = MarketStructureAnalyzer(confidence_threshold=0.5)
    all_events = analyzer.get_market_events(structure)
    
    print(f"\n📊 EVENT TIMING ANALYSIS")
    print("-" * 40)
    timing_analysis = analyze_event_timing(all_events)
    
    if 'error' not in timing_analysis:
        print(f"Event Frequency: {timing_analysis['event_frequency']}")
        if timing_analysis['time_between_events']:
            avg_time = np.mean(timing_analysis['time_between_events'])
            print(f"Average time between events: {avg_time:.1f} hours")
        
        if timing_analysis['consecutive_events']:
            print(f"Consecutive events: {timing_analysis['consecutive_events']}")
        
        if timing_analysis['potential_issues']:
            print(f"⚠️  Timing Issues:")
            for issue in timing_analysis['potential_issues']:
                print(f"    - {issue}")
    
    print(f"\n📊 PRICE LEVEL ANALYSIS")
    print("-" * 40)
    price_analysis = analyze_price_levels(all_events)
    
    if 'error' not in price_analysis:
        print(f"Level Effectiveness: {price_analysis['level_effectiveness']}")
        
        if price_analysis['price_clusters']:
            print(f"Price Clusters: {len(price_analysis['price_clusters'])} clusters found")
            for cluster in price_analysis['price_clusters'][:3]:  # Show first 3
                print(f"  - {cluster['count']} events around {cluster['price_range']}")
        
        if price_analysis['potential_issues']:
            print(f"⚠️  Price Issues:")
            for issue in price_analysis['potential_issues']:
                print(f"    - {issue}")
    
    print(f"\n📊 TREND CONSISTENCY ANALYSIS")
    print("-" * 40)
    trend_analysis = analyze_trend_consistency(all_events)
    
    if 'error' not in trend_analysis:
        print(f"Trend Consistency Score: {trend_analysis['consistency_score']:.2f}")
        print(f"Trend Changes: {len(trend_analysis['trend_changes'])}")
        print(f"Trend Reversals: {len(trend_analysis['trend_reversals'])}")
        
        if trend_analysis['potential_issues']:
            print(f"⚠️  Trend Issues:")
            for issue in trend_analysis['potential_issues']:
                print(f"    - {issue}")
    
    print(f"\n📊 CONFIDENCE ANALYSIS")
    print("-" * 40)
    confidence_analysis = analyze_confidence_distribution(all_events)
    
    if 'error' not in confidence_analysis:
        print(f"Confidence Distribution: {confidence_analysis['confidence_ranges']}")
        print(f"Low Confidence Events: {len(confidence_analysis['low_confidence_events'])}")
        
        if confidence_analysis['confidence_issues']:
            print(f"⚠️  Confidence Issues:")
            for issue in confidence_analysis['confidence_issues']:
                print(f"    - {issue}")
    
    # Overall improvement recommendations
    print(f"\n💡 DETAILED IMPROVEMENT RECOMMENDATIONS")
    print("=" * 70)
    
    total_issues = 0
    if 'error' not in timing_analysis:
        total_issues += len(timing_analysis['potential_issues'])
    if 'error' not in price_analysis:
        total_issues += len(price_analysis['potential_issues'])
    if 'error' not in trend_analysis:
        total_issues += len(trend_analysis['potential_issues'])
    if 'error' not in confidence_analysis:
        total_issues += len(confidence_analysis['confidence_issues'])
    
    if total_issues == 0:
        print("✅ EXCELLENT: No specific improvement areas identified!")
        print("   The current implementation is working very well.")
    elif total_issues <= 3:
        print("🟡 GOOD: Minor improvements can be made:")
        print("   Focus on the specific issues identified above.")
    else:
        print("🔴 MODERATE IMPROVEMENT NEEDED:")
        print("   Several areas identified for enhancement.")
    
    # Specific recommendations
    print(f"\n🎯 SPECIFIC IMPROVEMENTS:")
    
    # Timing improvements
    if 'error' not in timing_analysis and timing_analysis['potential_issues']:
        print("  1. Event Timing:")
        for issue in timing_analysis['potential_issues']:
            if "frequent" in issue.lower():
                print("     - Implement minimum time gap between events")
            elif "gaps" in issue.lower():
                print("     - Add intermediate event detection for long periods")
            elif "consecutive" in issue.lower():
                print("     - Add event consolidation logic")
    
    # Price improvements
    if 'error' not in price_analysis and price_analysis['potential_issues']:
        print("  2. Price Level Management:")
        for issue in price_analysis['potential_issues']:
            if "clusters" in issue.lower():
                print("     - Implement price clustering detection and consolidation")
            elif "level" in issue.lower():
                print("     - Strengthen level validation rules")
    
    # Trend improvements
    if 'error' not in trend_analysis and trend_analysis['potential_issues']:
        print("  3. Trend Consistency:")
        for issue in trend_analysis['potential_issues']:
            if "reversals" in issue.lower():
                print("     - Add trend reversal validation")
            elif "changes" in issue.lower():
                print("     - Implement trend change frequency limits")
    
    # Confidence improvements
    if 'error' not in confidence_analysis and confidence_analysis['confidence_issues']:
        print("  4. Confidence Management:")
        for issue in confidence_analysis['confidence_issues']:
            if "threshold" in issue.lower():
                print("     - Adjust confidence thresholds based on market conditions")
            elif "validation" in issue.lower():
                print("     - Enhance validation rules for higher confidence")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total Events: {len(all_events)}")
    print(f"  BOS Events: {len([e for e in all_events if e.event_type.value == 'BOS'])}")
    print(f"  CHOCH Events: {len([e for e in all_events if e.event_type.value == 'CHOCH'])}")
    print(f"  Issues Identified: {total_issues}")
    
    if total_issues == 0:
        print("  Status: ✅ EXCELLENT - No improvements needed")
    elif total_issues <= 3:
        print("  Status: 🟡 GOOD - Minor improvements possible")
    else:
        print("  Status: 🔴 MODERATE - Several improvements recommended")

if __name__ == "__main__":
    main()
