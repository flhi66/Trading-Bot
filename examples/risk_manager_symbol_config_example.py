"""
Example: Using Flexible Symbol Configuration with RiskManager

This example demonstrates how to use the new flexible symbol configuration
system for different brokers and symbols.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_manager import RiskManager
from config.symbol_config import get_symbol_config, get_broker_specific_config


def example_basic_usage():
    """Example of basic symbol configuration usage."""
    print("=== Basic Symbol Configuration Example ===")
    
    # Create RiskManager with default symbol configuration
    risk_manager = RiskManager()
    
    # Test different symbols
    symbols = ['EURUSD', 'USDJPY', 'XAUUSD', 'BTCUSD', 'SPX500']
    
    for symbol in symbols:
        pip_value = risk_manager._get_pip_value(symbol, 1.0)
        config = risk_manager.get_symbol_config(symbol)
        print(f"{symbol}: pip_value={pip_value:.6f}, decimals={config['pip_decimals']}")


def example_custom_configuration():
    """Example of custom symbol configuration."""
    print("\n=== Custom Symbol Configuration Example ===")
    
    # Create custom symbol configuration
    custom_config = {
        'EURUSD': {'pip_decimals': 4, 'custom_pip_value': None},
        'XAUUSD': {'pip_decimals': 2, 'custom_pip_value': 0.05},  # Custom pip value
        'BTCUSD': {'pip_decimals': 2, 'custom_pip_value': 5.0},   # Custom pip value
        'DEFAULT': {'pip_decimals': 4, 'custom_pip_value': None}
    }
    
    # Create RiskManager with custom configuration
    risk_manager = RiskManager(symbol_config=custom_config)
    
    # Test custom symbols
    symbols = ['EURUSD', 'XAUUSD', 'BTCUSD', 'UNKNOWN_SYMBOL']
    
    for symbol in symbols:
        pip_value = risk_manager._get_pip_value(symbol, 1.0)
        config = risk_manager.get_symbol_config(symbol)
        print(f"{symbol}: pip_value={pip_value:.6f}, decimals={config['pip_decimals']}")


def example_broker_specific_configuration():
    """Example of broker-specific configuration."""
    print("\n=== Broker-Specific Configuration Example ===")
    
    # Get broker-specific configuration
    mt4_config = get_broker_specific_config('MT4')
    
    # Create RiskManager with broker-specific configuration
    risk_manager = RiskManager(symbol_config=mt4_config)
    
    # Test broker-specific symbols
    symbols = ['XAUUSD', 'BTCUSD', 'SPX500']
    
    for symbol in symbols:
        pip_value = risk_manager._get_pip_value(symbol, 1.0)
        config = risk_manager.get_symbol_config(symbol)
        print(f"MT4 {symbol}: pip_value={pip_value:.6f}, decimals={config['pip_decimals']}")


def example_dynamic_symbol_addition():
    """Example of dynamically adding new symbols."""
    print("\n=== Dynamic Symbol Addition Example ===")
    
    # Create RiskManager
    risk_manager = RiskManager()
    
    # Add new symbols dynamically
    risk_manager.add_symbol_config('DOGEUSD', 4, 0.0001)
    risk_manager.add_symbol_config('SHIBUSD', 6, 0.000001)
    risk_manager.add_symbol_config('CUSTOM_INDEX', 1, 0.5)
    
    # Test new symbols
    new_symbols = ['DOGEUSD', 'SHIBUSD', 'CUSTOM_INDEX']
    
    for symbol in new_symbols:
        pip_value = risk_manager._get_pip_value(symbol, 1.0)
        config = risk_manager.get_symbol_config(symbol)
        print(f"{symbol}: pip_value={pip_value:.6f}, decimals={config['pip_decimals']}")


def example_risk_calculation_with_different_symbols():
    """Example of risk calculations with different symbols."""
    print("\n=== Risk Calculation with Different Symbols Example ===")
    
    # Create RiskManager
    risk_manager = RiskManager()
    
    # Test risk calculations for different symbols
    test_cases = [
        ('EURUSD', 1.1000, 'BUY', 0.001, 2.0),
        ('USDJPY', 110.00, 'SELL', 0.1, 2.0),
        ('XAUUSD', 1800.0, 'BUY', 2.0, 2.0),
        ('BTCUSD', 50000.0, 'BUY', 100.0, 2.0),
    ]
    
    for symbol, entry_price, direction, atr_value, rr_ratio in test_cases:
        result = risk_manager.compute_stop_and_target_from_atr(
            entry_price=entry_price,
            direction=direction,
            atr_value=atr_value,
            reward_risk_ratio=rr_ratio,
            symbol=symbol
        )
        
        if result:
            stop_loss, take_profit = result
            pip_value = risk_manager._get_pip_value(symbol, entry_price)
            risk_pips = abs(entry_price - stop_loss) / pip_value
            
            print(f"{symbol}: Entry={entry_price}, SL={stop_loss:.5f}, TP={take_profit:.5f}, Risk={risk_pips:.1f} pips")
        else:
            print(f"{symbol}: Failed to calculate stops")


if __name__ == "__main__":
    example_basic_usage()
    example_custom_configuration()
    example_broker_specific_configuration()
    example_dynamic_symbol_addition()
    example_risk_calculation_with_different_symbols()
