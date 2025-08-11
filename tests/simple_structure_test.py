import pandas as pd
import numpy as np
from utils.structure_trend_plotter import StructureTrendPlotter

def create_test_data():
    """Create simple test data"""
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    
    # Create trending price data
    base_price = 2000
    trend = np.linspace(0, 100, 100)  # Upward trend
    noise = np.random.normal(0, 5, 100)  # Random noise
    
    prices = base_price + trend + noise
    
    # Create OHLC data
    df = pd.DataFrame({
        'Open': prices + np.random.normal(0, 2, 100),
        'High': prices + np.abs(np.random.normal(2, 1, 100)),
        'Low': prices - np.abs(np.random.normal(2, 1, 100)),
        'Close': prices + np.random.normal(0, 1, 100),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
    df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
    
    return df

def main():
    print("🧪 Testing Structure & Trend Analysis with synthetic data...")
    
    # Create test data
    test_data = create_test_data()
    print(f"✅ Created test data with {len(test_data)} candles")
    print(f"   Price range: ${test_data['Low'].min():.2f} - ${test_data['High'].max():.2f}")
    
    # Initialize plotter
    plotter = StructureTrendPlotter()
    
    try:
        # Test Chart 1: Trend Detection
        print("\n-> Testing Chart 1: Trend Detection")
        fig1 = plotter.plot_trend_detection(test_data, "TEST", "1H")
        print("✅ Chart 1 created successfully")
        fig1.show()
        
        # Test Chart 2: Structure Analysis  
        print("\n-> Testing Chart 2: Structure Analysis")
        fig2 = plotter.plot_structure_analysis(test_data, "TEST", "1H")
        print("✅ Chart 2 created successfully")
        fig2.show()
        
        print("\n🎉 All charts generated successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()