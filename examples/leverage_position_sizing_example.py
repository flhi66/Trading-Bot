"""
Example: Improved Leverage and Position Sizing with Currency Conversion

This example demonstrates how the new currency conversion logic works
for accurate leverage calculations across different symbols.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_manager import RiskManager


def example_currency_conversion_rates():
    """Example showing currency conversion rates for different symbols."""
    print("=== Currency Conversion Rates ===")
    
    # Create RiskManager with USD account
    risk_manager = RiskManager(account_currency="USD")
    
    # Test different symbols and their conversion rates
    test_cases = [
        ("EURUSD", 1.1000),    # EUR base, USD quote
        ("USDJPY", 110.00),    # USD base, JPY quote
        ("GBPUSD", 1.2500),    # GBP base, USD quote
        ("XAUUSD", 1800.0),    # Gold base, USD quote
        ("BTCUSD", 50000.0),   # BTC base, USD quote
    ]
    
    print(f"Account Currency: {risk_manager.account_currency}")
    print()
    
    for symbol, entry_price in test_cases:
        conversion_rate = risk_manager._get_currency_conversion_rate(symbol, entry_price)
        print(f"{symbol} @ {entry_price}: Conversion Rate = {conversion_rate:.6f}")


def example_leverage_calculation_comparison():
    """Example comparing old vs new leverage calculation methods."""
    print("\n=== Leverage Calculation Comparison ===")
    
    # Test parameters
    account_balance = 10000.0  # $10,000 USD account
    max_leverage = 10.0
    entry_price = 1.1000
    symbol = "EURUSD"
    
    print(f"Account Balance: ${account_balance:,.2f}")
    print(f"Max Leverage: {max_leverage}x")
    print(f"Entry Price: {entry_price}")
    print(f"Symbol: {symbol}")
    print()
    
    # Old method (incorrect)
    old_max_position = account_balance * max_leverage / entry_price
    old_leverage = old_max_position * entry_price / account_balance
    
    # New method (correct)
    risk_manager = RiskManager(account_currency="USD")
    conversion_rate = risk_manager._get_currency_conversion_rate(symbol, entry_price)
    new_max_position = account_balance * max_leverage / (entry_price * conversion_rate)
    new_leverage = new_max_position * entry_price * conversion_rate / account_balance
    
    print("--- Old Method (Incorrect) ---")
    print(f"Max Position: {old_max_position:.2f} units")
    print(f"Calculated Leverage: {old_leverage:.2f}x")
    print()
    
    print("--- New Method (Correct) ---")
    print(f"Conversion Rate: {conversion_rate:.6f}")
    print(f"Max Position: {new_max_position:.2f} units")
    print(f"Calculated Leverage: {new_leverage:.2f}x")
    print()


def example_different_symbols_leverage():
    """Example showing leverage calculations for different symbols."""
    print("\n=== Leverage Calculations for Different Symbols ===")
    
    # Create RiskManager
    risk_manager = RiskManager(
        account_currency="USD",
        max_leverage=10.0,
        risk_per_trade=0.02
    )
    
    # Test parameters
    account_balance = 10000.0
    atr_value = 0.0010
    rr_ratio = 2.0
    
    # Test different symbols
    test_cases = [
        ("EURUSD", 1.1000, "BUY"),    # Major pair
        ("USDJPY", 110.00, "SELL"),   # JPY pair
        ("GBPUSD", 1.2500, "BUY"),    # GBP pair
        ("XAUUSD", 1800.0, "BUY"),    # Gold
        ("BTCUSD", 50000.0, "BUY"),   # Crypto
    ]
    
    print(f"Account Balance: ${account_balance:,.2f}")
    print(f"Max Leverage: {risk_manager.max_leverage}x")
    print()
    
    for symbol, entry_price, direction in test_cases:
        print(f"--- {symbol} @ {entry_price} ---")
        
        # Calculate realistic trade setup
        realistic_setup = risk_manager.compute_realistic_trade_setup(
            market_price=entry_price,
            direction=direction,
            atr_value=atr_value,
            reward_risk_ratio=rr_ratio,
            symbol=symbol
        )
        
        if realistic_setup:
            actual_entry_price, stop_loss, take_profit = realistic_setup
            
            # Calculate position size
            risk_amount = risk_manager.risk_amount_for_balance(account_balance)
            position_size = risk_manager.calculate_position_size(
                entry_price=actual_entry_price,
                stop_loss=stop_loss,
                risk_amount=risk_amount,
                account_balance=account_balance,
                symbol=symbol
            )
            
            # Calculate leverage metrics
            conversion_rate = risk_manager._get_currency_conversion_rate(symbol, actual_entry_price)
            position_value_usd = position_size * actual_entry_price * conversion_rate
            actual_leverage = position_value_usd / account_balance
            
            print(f"Entry Price: {actual_entry_price:.5f}")
            print(f"Position Size: {position_size:.2f} units")
            print(f"Position Value: ${position_value_usd:,.2f} USD")
            print(f"Actual Leverage: {actual_leverage:.2f}x")
            print(f"Conversion Rate: {conversion_rate:.6f}")
            print()


def example_usd_vs_non_usd_accounts():
    """Example comparing USD vs non-USD account currencies."""
    print("\n=== USD vs Non-USD Account Comparison ===")
    
    # Test parameters
    account_balance = 10000.0
    entry_price = 1.1000
    symbol = "EURUSD"
    
    print(f"Account Balance: {account_balance:,.2f}")
    print(f"Entry Price: {entry_price}")
    print(f"Symbol: {symbol}")
    print()
    
    # USD Account
    usd_risk_manager = RiskManager(account_currency="USD")
    usd_conversion_rate = usd_risk_manager._get_currency_conversion_rate(symbol, entry_price)
    usd_max_position = account_balance * usd_risk_manager.max_leverage / (entry_price * usd_conversion_rate)
    usd_leverage = usd_max_position * entry_price * usd_conversion_rate / account_balance
    
    print("--- USD Account ---")
    print(f"Conversion Rate: {usd_conversion_rate:.6f}")
    print(f"Max Position: {usd_max_position:.2f} units")
    print(f"Actual Leverage: {usd_leverage:.2f}x")
    print()
    
    # EUR Account (example)
    eur_risk_manager = RiskManager(account_currency="EUR")
    eur_conversion_rate = eur_risk_manager._get_currency_conversion_rate(symbol, entry_price)
    eur_max_position = account_balance * eur_risk_manager.max_leverage / (entry_price * eur_conversion_rate)
    eur_leverage = eur_max_position * entry_price * eur_conversion_rate / account_balance
    
    print("--- EUR Account ---")
    print(f"Conversion Rate: {eur_conversion_rate:.6f}")
    print(f"Max Position: {eur_max_position:.2f} units")
    print(f"Actual Leverage: {eur_leverage:.2f}x")
    print()


def example_leverage_impact_analysis():
    """Example showing how leverage limits affect position sizing."""
    print("\n=== Leverage Impact Analysis ===")
    
    # Test parameters
    account_balance = 10000.0
    entry_price = 1.1000
    symbol = "EURUSD"
    direction = "BUY"
    atr_value = 0.0010
    rr_ratio = 2.0
    
    # Test different leverage limits
    leverage_levels = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    print(f"Account Balance: ${account_balance:,.2f}")
    print(f"Entry Price: {entry_price}")
    print(f"Symbol: {symbol}")
    print()
    
    for max_leverage in leverage_levels:
        risk_manager = RiskManager(
            account_currency="USD",
            max_leverage=max_leverage,
            risk_per_trade=0.02
        )
        
        # Calculate realistic trade setup
        realistic_setup = risk_manager.compute_realistic_trade_setup(
            market_price=entry_price,
            direction=direction,
            atr_value=atr_value,
            reward_risk_ratio=rr_ratio,
            symbol=symbol
        )
        
        if realistic_setup:
            actual_entry_price, stop_loss, take_profit = realistic_setup
            
            # Calculate position size
            risk_amount = risk_manager.risk_amount_for_balance(account_balance)
            position_size = risk_manager.calculate_position_size(
                entry_price=actual_entry_price,
                stop_loss=stop_loss,
                risk_amount=risk_amount,
                account_balance=account_balance,
                symbol=symbol
            )
            
            # Calculate leverage metrics
            conversion_rate = risk_manager._get_currency_conversion_rate(symbol, actual_entry_price)
            position_value_usd = position_size * actual_entry_price * conversion_rate
            actual_leverage = position_value_usd / account_balance
            
            print(f"Max Leverage {max_leverage:3.0f}x: Position Size={position_size:8.2f}, "
                  f"Actual Leverage={actual_leverage:5.2f}x, Value=${position_value_usd:8,.2f}")


if __name__ == "__main__":
    example_currency_conversion_rates()
    example_leverage_calculation_comparison()
    example_different_symbols_leverage()
    example_usd_vs_non_usd_accounts()
    example_leverage_impact_analysis()
