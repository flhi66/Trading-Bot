"""
Example: Spread and Slippage Modeling in RiskManager

This example demonstrates how the new spread and slippage modeling works
for realistic trade execution costs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_manager import RiskManager


def example_basic_spread_slippage():
    """Example of basic spread and slippage modeling."""
    print("=== Basic Spread and Slippage Modeling ===")
    
    # Create RiskManager with spread and slippage settings
    risk_manager = RiskManager(
        spread_pips=1.5,      # 1.5 pip spread
        slippage_buffer=0.5,  # 0.5 pip slippage
        atr_multiplier=2.0
    )
    
    # Test parameters
    market_price = 1.1000  # EURUSD market price
    direction = "BUY"
    atr_value = 0.0010    # 10 pips ATR
    rr_ratio = 2.0
    symbol = "EURUSD"
    
    print(f"Market Price: {market_price}")
    print(f"Direction: {direction}")
    print(f"ATR: {atr_value}")
    print(f"Risk-Reward: {rr_ratio}")
    print(f"Spread: {risk_manager.spread_pips} pips")
    print(f"Slippage: {risk_manager.slippage_buffer} pips")
    print()
    
    # Method 1: Step-by-step approach
    print("--- Step-by-Step Approach ---")
    
    # Step 1: Calculate stops based on market price
    stop_result = risk_manager.compute_stop_and_target_from_atr(
        market_price=market_price,
        direction=direction,
        atr_value=atr_value,
        reward_risk_ratio=rr_ratio,
        symbol=symbol
    )
    
    if stop_result:
        stop_loss, take_profit = stop_result
        print(f"Market-based stops: SL={stop_loss:.5f}, TP={take_profit:.5f}")
        
        # Step 2: Apply spread and slippage
        entry_price, adj_stop_loss, adj_take_profit = risk_manager.apply_spread_and_slippage(
            market_price=market_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=direction,
            symbol=symbol
        )
        
        print(f"Realistic execution: Entry={entry_price:.5f}, SL={adj_stop_loss:.5f}, TP={adj_take_profit:.5f}")
    
    print()
    
    # Method 2: All-in-one approach
    print("--- All-in-One Approach ---")
    
    realistic_setup = risk_manager.compute_realistic_trade_setup(
        market_price=market_price,
        direction=direction,
        atr_value=atr_value,
        reward_risk_ratio=rr_ratio,
        symbol=symbol
    )
    
    if realistic_setup:
        entry_price, stop_loss, take_profit = realistic_setup
        print(f"Complete setup: Entry={entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}")


def example_different_symbols():
    """Example with different symbols and their spread characteristics."""
    print("\n=== Different Symbols and Spreads ===")
    
    # Create RiskManager with different spread settings
    risk_manager = RiskManager(
        spread_pips=2.0,      # 2 pip spread
        slippage_buffer=1.0,  # 1 pip slippage
        atr_multiplier=2.0
    )
    
    # Test different symbols
    test_cases = [
        ("EURUSD", 1.1000, "BUY", 0.0010, 2.0),    # Major pair
        ("USDJPY", 110.00, "SELL", 0.10, 2.0),     # JPY pair
        ("XAUUSD", 1800.0, "BUY", 2.0, 2.0),       # Gold
        ("BTCUSD", 50000.0, "BUY", 100.0, 2.0),    # Crypto
    ]
    
    for symbol, market_price, direction, atr_value, rr_ratio in test_cases:
        print(f"\n--- {symbol} ---")
        print(f"Market Price: {market_price}")
        print(f"Direction: {direction}")
        print(f"ATR: {atr_value}")
        
        realistic_setup = risk_manager.compute_realistic_trade_setup(
            market_price=market_price,
            direction=direction,
            atr_value=atr_value,
            reward_risk_ratio=rr_ratio,
            symbol=symbol
        )
        
        if realistic_setup:
            entry_price, stop_loss, take_profit = realistic_setup
            
            # Calculate costs
            pip_value = risk_manager._get_pip_value(symbol, market_price)
            spread_cost = risk_manager.spread_pips * pip_value
            slippage_cost = risk_manager.slippage_buffer * pip_value
            total_cost = spread_cost + slippage_cost
            
            print(f"Entry: {entry_price:.5f}")
            print(f"Stop Loss: {stop_loss:.5f}")
            print(f"Take Profit: {take_profit:.5f}")
            print(f"Spread Cost: {spread_cost:.5f} ({risk_manager.spread_pips} pips)")
            print(f"Slippage Cost: {slippage_cost:.5f} ({risk_manager.slippage_buffer} pips)")
            print(f"Total Execution Cost: {total_cost:.5f}")


def example_spread_impact_analysis():
    """Example showing the impact of different spread levels."""
    print("\n=== Spread Impact Analysis ===")
    
    # Test parameters
    market_price = 1.1000
    direction = "BUY"
    atr_value = 0.0010
    rr_ratio = 2.0
    symbol = "EURUSD"
    
    # Test different spread levels
    spread_levels = [0.5, 1.0, 1.5, 2.0, 3.0]  # pips
    
    print(f"Market Price: {market_price}")
    print(f"Direction: {direction}")
    print(f"ATR: {atr_value}")
    print(f"Risk-Reward: {rr_ratio}")
    print()
    
    for spread_pips in spread_levels:
        risk_manager = RiskManager(
            spread_pips=spread_pips,
            slippage_buffer=0.5,
            atr_multiplier=2.0
        )
        
        realistic_setup = risk_manager.compute_realistic_trade_setup(
            market_price=market_price,
            direction=direction,
            atr_value=atr_value,
            reward_risk_ratio=rr_ratio,
            symbol=symbol
        )
        
        if realistic_setup:
            entry_price, stop_loss, take_profit = realistic_setup
            
            # Calculate metrics
            pip_value = risk_manager._get_pip_value(symbol, market_price)
            spread_cost = spread_pips * pip_value
            risk_distance = abs(entry_price - stop_loss)
            reward_distance = abs(take_profit - entry_price)
            effective_rr = reward_distance / risk_distance if risk_distance > 0 else 0
            
            print(f"Spread {spread_pips:3.1f} pips: Entry={entry_price:.5f}, "
                  f"Effective RR={effective_rr:.2f}, Cost={spread_cost:.5f}")


def example_buy_vs_sell_comparison():
    """Example comparing BUY vs SELL with spread impact."""
    print("\n=== BUY vs SELL Comparison ===")
    
    risk_manager = RiskManager(
        spread_pips=1.5,
        slippage_buffer=0.5,
        atr_multiplier=2.0
    )
    
    market_price = 1.1000
    atr_value = 0.0010
    rr_ratio = 2.0
    symbol = "EURUSD"
    
    print(f"Market Price: {market_price}")
    print(f"ATR: {atr_value}")
    print(f"Risk-Reward: {rr_ratio}")
    print()
    
    for direction in ["BUY", "SELL"]:
        print(f"--- {direction} Trade ---")
        
        realistic_setup = risk_manager.compute_realistic_trade_setup(
            market_price=market_price,
            direction=direction,
            atr_value=atr_value,
            reward_risk_ratio=rr_ratio,
            symbol=symbol
        )
        
        if realistic_setup:
            entry_price, stop_loss, take_profit = realistic_setup
            
            # Calculate metrics
            pip_value = risk_manager._get_pip_value(symbol, market_price)
            spread_cost = risk_manager.spread_pips * pip_value
            slippage_cost = risk_manager.slippage_buffer * pip_value
            
            print(f"Entry: {entry_price:.5f}")
            print(f"Stop Loss: {stop_loss:.5f}")
            print(f"Take Profit: {take_profit:.5f}")
            print(f"Spread Cost: {spread_cost:.5f}")
            print(f"Slippage Cost: {slippage_cost:.5f}")
            print()


if __name__ == "__main__":
    example_basic_spread_slippage()
    example_different_symbols()
    example_spread_impact_analysis()
    example_buy_vs_sell_comparison()
