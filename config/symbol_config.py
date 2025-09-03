"""
Symbol Configuration for Risk Manager

This file contains symbol-specific configurations for pip values and decimal places.
Users can customize these values based on their broker's specifications.

Usage:
    from config.symbol_config import get_symbol_config
    risk_manager = RiskManager(symbol_config=get_symbol_config())
"""

def get_symbol_config() -> dict:
    """
    Get symbol configuration dictionary.
    
    Returns:
        Dictionary with symbol configurations
    """
    return {
        # Major pairs (4 decimal places, 1 pip = 0.0001)
        'EURUSD': {'pip_decimals': 4, 'custom_pip_value': None},
        'GBPUSD': {'pip_decimals': 4, 'custom_pip_value': None},
        'AUDUSD': {'pip_decimals': 4, 'custom_pip_value': None},
        'NZDUSD': {'pip_decimals': 4, 'custom_pip_value': None},
        'USDCAD': {'pip_decimals': 4, 'custom_pip_value': None},
        'USDCHF': {'pip_decimals': 4, 'custom_pip_value': None},
        
        # JPY pairs (2 decimal places, 1 pip = 0.01)
        'USDJPY': {'pip_decimals': 2, 'custom_pip_value': None},
        'EURJPY': {'pip_decimals': 2, 'custom_pip_value': None},
        'GBPJPY': {'pip_decimals': 2, 'custom_pip_value': None},
        'AUDJPY': {'pip_decimals': 2, 'custom_pip_value': None},
        'NZDJPY': {'pip_decimals': 2, 'custom_pip_value': None},
        'CADJPY': {'pip_decimals': 2, 'custom_pip_value': None},
        'CHFJPY': {'pip_decimals': 2, 'custom_pip_value': None},
        
        # Gold (pip value can vary by broker - adjust as needed)
        'XAUUSD': {'pip_decimals': 2, 'custom_pip_value': 0.01},  # Adjust based on your broker
        'GOLD': {'pip_decimals': 2, 'custom_pip_value': 0.01},
        
        # Silver (3 decimal places)
        'XAGUSD': {'pip_decimals': 3, 'custom_pip_value': None},
        'SILVER': {'pip_decimals': 3, 'custom_pip_value': None},
        
        # Crypto (pip values vary by exchange - adjust as needed)
        'BTCUSD': {'pip_decimals': 2, 'custom_pip_value': 1.0},   # 1 USD per pip
        'ETHUSD': {'pip_decimals': 2, 'custom_pip_value': 0.1},   # 0.1 USD per pip
        'LTCUSD': {'pip_decimals': 2, 'custom_pip_value': 0.1},
        'ADAUSD': {'pip_decimals': 4, 'custom_pip_value': 0.0001},
        'DOTUSD': {'pip_decimals': 3, 'custom_pip_value': 0.001},
        
        # Indices (pip values vary by broker - adjust as needed)
        'SPX500': {'pip_decimals': 1, 'custom_pip_value': 0.1},   # 0.1 points per pip
        'NAS100': {'pip_decimals': 1, 'custom_pip_value': 0.1},   # 0.1 points per pip
        'US30': {'pip_decimals': 1, 'custom_pip_value': 1.0},     # 1 point per pip
        'UK100': {'pip_decimals': 1, 'custom_pip_value': 0.1},    # 0.1 points per pip
        'GER30': {'pip_decimals': 1, 'custom_pip_value': 0.1},    # 0.1 points per pip
        
        # Commodities (adjust based on your broker)
        'WTI': {'pip_decimals': 2, 'custom_pip_value': 0.01},     # Crude oil
        'BRENT': {'pip_decimals': 2, 'custom_pip_value': 0.01},   # Brent oil
        'NATGAS': {'pip_decimals': 3, 'custom_pip_value': 0.001}, # Natural gas
        
        # Default fallback
        'DEFAULT': {'pip_decimals': 4, 'custom_pip_value': None}
    }


def get_broker_specific_config(broker_name: str) -> dict:
    """
    Get broker-specific symbol configurations.
    
    Args:
        broker_name: Name of the broker (e.g., 'MT4', 'MT5', 'cTrader')
        
    Returns:
        Broker-specific symbol configuration
    """
    base_config = get_symbol_config()
    
    if broker_name.upper() == 'MT4':
        # MT4 specific adjustments
        base_config['XAUUSD']['custom_pip_value'] = 0.01
        base_config['BTCUSD']['custom_pip_value'] = 1.0
        base_config['SPX500']['custom_pip_value'] = 0.1
        
    elif broker_name.upper() == 'MT5':
        # MT5 specific adjustments
        base_config['XAUUSD']['custom_pip_value'] = 0.01
        base_config['BTCUSD']['custom_pip_value'] = 1.0
        base_config['SPX500']['custom_pip_value'] = 0.1
        
    elif broker_name.upper() == 'CTRADER':
        # cTrader specific adjustments
        base_config['XAUUSD']['custom_pip_value'] = 0.01
        base_config['BTCUSD']['custom_pip_value'] = 1.0
        base_config['SPX500']['custom_pip_value'] = 0.1
        
    return base_config


# Example usage:
if __name__ == "__main__":
    # Basic usage
    config = get_symbol_config()
    print("EURUSD pip value:", 10 ** (-config['EURUSD']['pip_decimals']))
    
    # Broker-specific usage
    mt4_config = get_broker_specific_config('MT4')
    print("MT4 XAUUSD pip value:", mt4_config['XAUUSD']['custom_pip_value'])
