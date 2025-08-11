"""
Comprehensive Pattern Recognition Test
Implements all features from the 4.0 Patterns Identification notebook
"""

import pandas as pd
import numpy as np
import webbrowser
import os
from datetime import datetime, timedelta

# Import our pattern recognition modules
from core.pattern_recognition import (
    PivotPointDetector, 
    PiecewiseLinearizer, 
    DistanceMetrics, 
    PatternRecognizer
)
from core.pattern_clustering import PatternAnalyzer
from utils.pattern_plotter import PatternPlotter
from core.data_loader import load_and_resample

def main():
    print("🔍 Pattern Recognition Analysis Starting...")
    print("=" * 60)
    
    # === Step 1: Load Data ===
    print("\n📊 Step 1: Loading Data...")
    symbol = "XAUUSD_H1.csv"
    resampled = load_and_resample(f"data/{symbol}")
    h1_data = resampled.get("1H")
    
    if h1_data is None or h1_data.empty:
        print(f"❌ ERROR: No data loaded for the '1H' timeframe.")
        return
    
    print(f"✅ Loaded {len(h1_data)} data points for {symbol}")
    print(f"📅 Date range: {h1_data.index[0]} to {h1_data.index[-1]}")
    
    # === Step 2: Pivot Point Detection ===
    print("\n🎯 Step 2: Pivot Point Detection...")
    # Use enhanced pivot detection with better parameters
    pivot_detector = PivotPointDetector(
        threshold_percentage=0.5,  # Increased from 0.3% to 0.5%
        min_duration='1H',
        use_multi_timeframe=True,
        confirmation_periods=5  # 5-period confirmation for more reliable pivots
    )
    
    # Find pivot points
    pivot_points = pivot_detector.find_pivot_points(h1_data, 'Close')
    print(f"✅ Found {len(pivot_points)} pivot points")
    
    # Show some pivot points
    print("\n📈 Sample Pivot Points:")
    for i, point in enumerate(pivot_points[:5]):
        print(f"  {i+1}. {point.pivot_type.upper()} @ {point.price:.2f} "
              f"({point.percentage_change:.1%}) - {point.timestamp}")
    
    # === Step 3: Price Movements ===
    print("\n📊 Step 3: Price Movements Analysis...")
    movements = pivot_detector.find_movements(h1_data, 'Close')
    
    print(f"✅ Uptrends: {len(movements['uptrends'])}")
    print(f"✅ Downtrends: {len(movements['downtrends'])}")
    
    if not movements['uptrends'].empty:
        print("\n📈 Sample Uptrends:")
        for i, (_, movement) in enumerate(movements['uptrends'].head(3).iterrows()):
            print(f"  {i+1}. {movement['start_timestamp']} → {movement['end_timestamp']} "
                  f"({movement['percentage_change']:.1%})")
    
    if not movements['downtrends'].empty:
        print("\n📉 Sample Downtrends:")
        for i, (_, movement) in enumerate(movements['downtrends'].head(3).iterrows()):
            print(f"  {i+1}. {movement['start_timestamp']} → {movement['end_timestamp']} "
                  f"({movement['percentage_change']:.1%})")
    
    # === Step 4: Piecewise Linearization ===
    print("\n📐 Step 4: Piecewise Linearization...")
    linearizer = PiecewiseLinearizer(timeframe='1H')
    
    # Extract patterns from different time periods
    pattern_periods = [
        ('2025-05-25', '2025-05-26', "May 25-26 Pattern"),
        ('2025-05-27', '2025-05-28', "May 27-28 Pattern"),
        ('2025-05-29', '2025-05-30', "May 29-30 Pattern")
    ]
    
    patterns = []
    pattern_names = []
    
    for start_date, end_date, name in pattern_periods:
        try:
            pattern = linearizer.linearize_movements(movements, start_date, end_date)
            if not pattern.empty:
                patterns.append(pattern)
                pattern_names.append(name)
                print(f"✅ {name}: {len(pattern)} points")
            else:
                print(f"⚠️  {name}: No data available")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    # === Step 5: Distance Metrics ===
    print("\n📏 Step 5: Distance Metrics...")
    distance_calculator = DistanceMetrics()
    
    if len(patterns) >= 2:
        print("\n📊 Pattern Distances:")
        
        # Calculate distances between all patterns
        n_patterns = len(patterns)
        distance_matrix = np.zeros((n_patterns, n_patterns))
        
        for i in range(n_patterns):
            for j in range(n_patterns):
                if i != j:
                    # Euclidean distance
                    try:
                        euclidean_dist = distance_calculator.euclidean_distance(
                            patterns[i].values, patterns[j].values
                        )
                        print(f"  {pattern_names[i]} ↔ {pattern_names[j]}:")
                        print(f"    Euclidean: {euclidean_dist:.5f}")
                    except Exception as e:
                        print(f"    Euclidean: Error - {e}")
                    
                    # DTW distance
                    try:
                        dtw_dist = distance_calculator.dtw_distance(
                            patterns[i].values, patterns[j].values
                        )
                        print(f"    DTW: {dtw_dist:.5f}")
                        distance_matrix[i][j] = dtw_dist
                    except Exception as e:
                        print(f"    DTW: Error - {e}")
        
        print(f"\n📊 Distance Matrix Shape: {distance_matrix.shape}")
    
    # === Step 6: Pattern Clustering ===
    print("\n🎯 Step 6: Pattern Clustering...")
    
    if len(patterns) >= 3:
        # Convert patterns to numpy arrays for clustering
        pattern_arrays = [p.values for p in patterns if not p.empty]
        
        if len(pattern_arrays) >= 2:
            # Create timestamps and prices for clustering
            timestamps = []
            prices = []
            for pattern in patterns:
                if not pattern.empty:
                    timestamps.append(pattern.index[0])
                    prices.append(pattern.iloc[-1])
            
            # Fit clustering
            n_clusters = min(3, len(pattern_arrays))
            analyzer = PatternAnalyzer(n_clusters=n_clusters)
            
            try:
                clusters = analyzer.fit_clustering(
                    pattern_arrays, 
                    timestamps, 
                    prices
                )
                
                print(f"✅ Created {len(clusters)} clusters")
                
                # Show cluster summary
                cluster_summary = analyzer.get_cluster_summary()
                if not cluster_summary.empty:
                    print("\n📊 Cluster Summary:")
                    print(cluster_summary.to_string(index=False))
                
            except Exception as e:
                print(f"❌ Clustering error: {e}")
        else:
            print("⚠️  Need at least 2 patterns for clustering")
    else:
        print("⚠️  Need at least 3 patterns for clustering")
    
    # === Step 7: Pattern Classification ===
    print("\n🏷️  Step 7: Pattern Classification...")
    
    if len(patterns) >= 2 and 'analyzer' in locals() and analyzer is not None:
        # Create some dummy labels for demonstration
        labels = ['uptrend', 'downtrend', 'sideways'][:len(patterns)]
        
        try:
            # Fit classification
            analyzer.fit_classification(pattern_arrays, labels)
            print(f"✅ Fitted classification model with {len(labels)} classes")
            
            # Test classification on first pattern
            if pattern_arrays:
                test_pattern = pattern_arrays[0]
                predicted_label = analyzer.classification.predict(test_pattern)
                probabilities = analyzer.classification.predict_proba(test_pattern)
                
                print(f"\n🔍 Classification Test:")
                print(f"  Pattern: {pattern_names[0]}")
                print(f"  Predicted: {predicted_label}")
                print(f"  Probabilities: {probabilities}")
                
        except Exception as e:
            print(f"❌ Classification error: {e}")
    else:
        print("⚠️  Need at least 2 patterns and analyzer for classification")
    
    # === Step 8: Create Visualizations ===
    print("\n🎨 Step 8: Creating Visualizations...")
    
    # Create output directory
    output_dir = "generated_pattern_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plotter = PatternPlotter()
    
    try:
        # 1. Enhanced Pivot Points Plot
        fig_pivot = plotter.plot_pivot_points(h1_data, pivot_points, 'Close', 
                                            f"Enhanced Pivot Points - {symbol}")
        pivot_file = os.path.join(output_dir, "enhanced_pivot_points.html")
        fig_pivot.write_html(pivot_file)
        print(f"✅ Enhanced Pivot Points plot saved: {pivot_file}")
        
        # 2. Comprehensive Pivot Analysis Plot (NEW)
        fig_comprehensive = plotter.plot_comprehensive_pivot_analysis(h1_data, pivot_points, 'Close',
                                                                    f"Comprehensive Pivot Analysis - {symbol}")
        comprehensive_file = os.path.join(output_dir, "comprehensive_pivot_analysis.html")
        fig_comprehensive.write_html(comprehensive_file)
        print(f"✅ Comprehensive Pivot Analysis plot saved: {comprehensive_file}")
        
        # 3. Movements Plot
        fig_movements = plotter.plot_movements(h1_data, movements, 'Close',
                                             f"Price Movements - {symbol}")
        movements_file = os.path.join(output_dir, "price_movements.html")
        fig_movements.write_html(movements_file)
        print(f"✅ Movements plot saved: {movements_file}")
        
        # 4. Pattern Comparison Plot
        if len(patterns) >= 2:
            fig_patterns = plotter.plot_patterns(patterns, pattern_names,
                                               "Pattern Comparison")
            patterns_file = os.path.join(output_dir, "pattern_comparison.html")
            fig_patterns.write_html(patterns_file)
            print(f"✅ Pattern comparison plot saved: {patterns_file}")
        
        # 5. Clusters Plot
        if 'clusters' in locals() and clusters:
            fig_clusters = plotter.plot_clusters(clusters, "Pattern Clusters")
            clusters_file = os.path.join(output_dir, "pattern_clusters.html")
            fig_clusters.write_html(clusters_file)
            print(f"✅ Clusters plot saved: {clusters_file}")
        
        # 6. Distance Matrix Plot
        if 'distance_matrix' in locals() and len(patterns) >= 2:
            fig_distance = plotter.plot_distance_matrix(patterns, distance_matrix, 
                                                     pattern_names, "Pattern Distance Matrix")
            distance_file = os.path.join(output_dir, "distance_matrix.html")
            fig_distance.write_html(distance_file)
            print(f"✅ Distance matrix plot saved: {distance_file}")
        
        print(f"\n🎉 All plots saved to: {os.path.abspath(output_dir)}")
        
        # Open plots in browser
        print("\n🌐 Opening plots in browser...")
        webbrowser.open(f"file://{os.path.abspath(comprehensive_file)}")
        webbrowser.open(f"file://{os.path.abspath(pivot_file)}")
        webbrowser.open(f"file://{os.path.abspath(movements_file)}")
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    
    # === Summary ===
    print("\n" + "=" * 60)
    print("🎯 PATTERN RECOGNITION ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📊 Data Points: {len(h1_data)}")
    print(f"🎯 Pivot Points: {len(pivot_points)}")
    print(f"📈 Uptrends: {len(movements['uptrends'])}")
    print(f"📉 Downtrends: {len(movements['downtrends'])}")
    print(f"📐 Patterns Extracted: {len(patterns)}")
    
    if 'clusters' in locals():
        print(f"🎯 Clusters Created: {len(clusters)}")
    
    print(f"\n📁 Plots saved to: {os.path.abspath(output_dir)}")
    print("✅ Analysis complete! Check the generated plots for visual insights.")

if __name__ == "__main__":
    main()
