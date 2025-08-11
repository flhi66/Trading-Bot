#!/usr/bin/env python3
"""
Test script for the BOS/CHOCH Backtester
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        from core.backtester import BOSCHOCHBacktester, BacktestVisualizer, Trade, BacktestResult
        print("✅ All backtester modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loader():
    """Test if data loader can be imported"""
    try:
        from core.data_loader import load_and_resample
        print("✅ Data loader imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Data loader import error: {e}")
        return False

def test_smart_money_concepts():
    """Test if smart money concepts can be imported"""
    try:
        from core.smart_money_concepts import MarketStructureAnalyzer, StructurePoint, SwingType, MarketEvent
        print("✅ Smart money concepts imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Smart money concepts import error: {e}")
        return False

def test_structure_builder():
    """Test if structure builder can be imported"""
    try:
        from core.structure_builder import get_market_analysis
        print("✅ Structure builder imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Structure builder import error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TESTING BOS/CHOCH BACKTESTER")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_data_loader,
        test_smart_money_concepts,
        test_structure_builder
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The backtester is ready to use.")
        print("\n🚀 To run the backtester:")
        print("   python backtester.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\n💡 Make sure you have installed all required dependencies:")
        print("   pip install -r requirements_backtester.txt")

if __name__ == "__main__":
    main()
